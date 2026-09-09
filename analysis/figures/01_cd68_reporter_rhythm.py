"""Figure 1: broad-period CD68 reporter rhythms in individual cells.

Each reusable panel can be drawn alone or in the canonical three-panel page::

    python analysis/figures/01_cd68_reporter_rhythm.py <run>
    ... --panels raster
    ... --panels period_peak_matrix
    ... --panels period_histogram
    ... --metrics area_px
    ... --fit-method mesa --significance-method lomb
    ... --detrend running_mean --detrend-window-hours 12
    ... --secondary-significance-method chi_square
    ... --multiple-testing bh
    ... --period-bins 2,6,12,18,24,30,36,42,48

The card performs a fresh Circadian Workbench fit so its estimator, rhythm
test, period range and detrending controls all follow the shared figure option
contract. One significance test defines the default verdict. Supplying a
secondary significance method opts into requiring both tests.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from _bundle import provenance_note
import matrix_ordering
from _metrics import describe
from _options import commas, numbers
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import category_strip, colour_bar, resolve_look
from panels import rhythms as rhythm_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


ORDER_ALIASES = {
    "principal_component": "pattern_rank",
    "pattern": "pattern_rank",
    "spectral": "spectral_rank",
    "onset": "displayed_onset_hours",
    "status": "rhythm_rank",
    "period": "period_hours",
    "significance": "q_value",
    "identity": "identity",
}

ORDER_LABELS = {
    "principal_component": "principal-component gradient",
    "pattern": "principal-component gradient",
    "spectral": "spectral continuum",
    "onset": "displayed onset",
    "status": "rhythm status",
    "period": "estimated period",
    "significance": "adjusted significance",
    "identity": "cell identity",
}


def _resolved_order(tokens: list[str]) -> list[str]:
    return matrix_ordering.resolve_order(
        tokens, aliases=ORDER_ALIASES, append=("period_hours",),
    )


def _order_label(tokens: list[str]) -> str:
    return matrix_ordering.order_label(tokens, ORDER_LABELS)


def _period_edges(
    search: list[float], bins: int, requested: list[float] | None,
) -> np.ndarray:
    low, high = map(float, search)
    if requested:
        edges = np.asarray(requested, dtype=float)
    else:
        if int(bins) < 2:
            raise SystemExit("--bins must be at least 2")
        edges = np.linspace(low, high, int(bins) + 1)
    if (edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all()
            or not np.all(np.diff(edges) > 0)):
        raise SystemExit(
            "--period-bins needs at least three increasing finite boundaries"
        )
    if edges[0] > low or edges[-1] < high:
        raise SystemExit(
            "--period-bins must cover the complete configured period search"
        )
    return edges


def _fit_cells(
    frame: pd.DataFrame,
    metric: str,
    resolved: dict,
    secondary_method: str | None,
) -> pd.DataFrame:
    """Fit periods and one or two corrected rhythm-test families."""
    params = resolved["params"]
    method = resolved["method"]
    primary_method = resolved["significance_method"]
    fit_args = dict(
        frame=frame[["identity", "hours", metric]],
        group_columns=["identity"],
        value_column=metric,
        params=params,
        method=method,
        detrend=resolved["detrend"],
        detrend_window_hours=resolved["detrend_window_hours"],
        min_observations=resolved["min_observations"],
        correction=resolved["multiple_testing"],
        min_cycles=resolved["min_cycles"],
    )
    fits = workbench.estimate_grouped_rhythms(
        **fit_args, significance_method=primary_method,
    )
    if fits.empty:
        return fits

    fits["period_estimation_method"] = method
    fits["primary_significance_method"] = primary_method
    fits["primary_test_status"] = fits["test_status"]
    fits["primary_p_value"] = fits["p_value"]
    fits["primary_q_value"] = fits["q_value"]
    fits["primary_significant"] = fits["significant"]

    secondary = None if secondary_method in (None, "") else str(secondary_method)
    if secondary is None:
        fits["secondary_significance_method"] = None
        fits["verdict_rule"] = f"{primary_method} only"
    else:
        try:
            secondary = workbench.resolve_significance_method(method, secondary)
        except ValueError as error:
            raise SystemExit(str(error)) from None
        if secondary == primary_method:
            raise SystemExit(
                "--secondary-significance-method must differ from "
                "--significance-method"
            )
        second = workbench.estimate_grouped_rhythms(
            **fit_args, significance_method=secondary,
        )[[
            "identity", "test_status", "p_value", "q_value", "significant",
            "significance_period_hours",
        ]].rename(columns={
            "test_status": "secondary_test_status",
            "p_value": "secondary_p_value",
            "q_value": "secondary_q_value",
            "significant": "secondary_significant",
            "significance_period_hours": "secondary_significance_period_hours",
        })
        fits = fits.merge(second, on="identity", how="left", validate="one_to_one")
        both_tested = (
            fits["primary_test_status"].eq("ok")
            & fits["secondary_test_status"].eq("ok")
        )
        fits["significant"] = (
            both_tested
            & fits["primary_significant"].fillna(False).astype(bool)
            & fits["secondary_significant"].fillna(False).astype(bool)
        )
        fits["test_status"] = np.where(both_tested, "ok", "not_tested")
        fits["rhythm_status"] = np.select(
            [~both_tested, fits["significant"]],
            ["not tested", "rhythmic"],
            default="not rhythmic",
        )
        fits["secondary_significance_method"] = secondary
        fits["verdict_rule"] = f"{primary_method} and {secondary}"

    fits["rhythmic"] = fits["significant"]
    return fits


def _statistics_table(fits: pd.DataFrame) -> pd.DataFrame:
    """One auditable row per cell and significance test."""
    shared = [
        "identity", "metric", "observations", "span_hours",
        "period_estimation_method", "period_hours", "period_error_hours",
        "estimate_status", "cycles_observed", "period_underdetermined",
        "period_at_search_edge", "correction", "alpha", "family_tests",
        "period_search_min_hours", "period_search_max_hours", "detrend",
        "detrend_window_hours", "detrend_polynomial_degree",
        "detrend_min_valid_fraction", "detrend_bandwidth_hours",
        "detrend_low_cut_hours", "detrend_high_cut_hours",
        "detrend_filter_order", "workbench_version",
        "analysis_parameters_json", "verdict_rule", "row_order_method",
        "row_order_keys_json",
    ]
    shared = [column for column in shared if column in fits]

    def rows_for(role: str, prefix: str, period_column: str) -> pd.DataFrame:
        table = fits[shared].copy()
        table["test_role"] = role
        table["significance_method"] = fits[f"{prefix}_significance_method"]
        table["test_status"] = fits[f"{prefix}_test_status"]
        table["p_value"] = fits[f"{prefix}_p_value"]
        table["q_value"] = fits[f"{prefix}_q_value"]
        table["test_significant"] = fits[f"{prefix}_significant"]
        table["significance_period_hours"] = fits[period_column]
        table["final_cell_significant"] = fits["significant"]
        table["final_rhythm_status"] = fits["rhythm_status"]
        return table

    tables = [rows_for("primary", "primary", "significance_period_hours")]
    if ("secondary_test_status" in fits
            and fits["secondary_significance_method"].notna().any()):
        tables.append(rows_for(
            "secondary", "secondary", "secondary_significance_period_hours",
        ))
    result = pd.concat(tables, ignore_index=True)
    leading = [
        "identity", "metric", "test_role", "significance_method",
        "test_status", "p_value", "q_value", "test_significant",
        "final_cell_significant", "final_rhythm_status",
        "period_estimation_method", "period_hours",
        "significance_period_hours",
    ]
    return result[[*leading, *[column for column in result if column not in leading]]]


@figure(
    number=1,
    slug="cd68-reporter-rhythm",
    summary="broad-period CD68 reporter rhythms, detected-cycle peaks and significant periods",
    title="{metric}: significant rhythms across the {search} h search",
    grammar="ordered trace raster above a period-by-peak matrix and significant-period histogram",
    reads=(Table("cell_frame.csv", module="measurement modules"),),
    panels=(
        Panel("raster", rhythm_panels.trace_raster,
              title="One detrended row per cell",
              min_width_inches=11.0, min_height_inches=7.0),
        Panel("period_peak_matrix", rhythm_panels.timing_by_period,
              title="Detected period by peak position within each cell's own cycle",
              min_width_inches=9.5, min_height_inches=4.0),
        Panel("period_histogram", rhythm_panels.significant_period_histogram,
              title="Estimated periods among significant cells only",
              min_width_inches=9.5, min_height_inches=4.0),
    ),
    options=(
        Option("metrics", default="corrected_mean", cast=str, metavar="COL",
               help="which measured cell-frame trace is tested and drawn"),
        Option("bins", default=12),
        Option("period_bins", default=None, cast=numbers),
        Option("trace_luts", default=None, cast=str, metavar="NAME",
               help="colour map the raster is drawn through"),
        Option("hour_ticks", default=None),
        Option("order", default=["principal_component"], cast=commas,
               metavar="COL,COL",
               help=(
                   "row order: principal_component (default), spectral, "
                   "onset,period, or period"
               )),
        Option("fit_method", default=None),
        Option("significance_method", default=None),
        Option("secondary_significance_method", default=None),
        Option("period_config", default={}),
        Option("period_min_hours", default=2.0),
        Option("period_max_hours", default=48.0),
        Option("rhythmic_alpha", default=0.05),
        Option(
            "multiple_testing",
            default="none",
            help=(
                "p-value correction: none (default/off), bh, bonferroni, "
                "or sidak"
            ),
        ),
        Option("min_observations", default=24),
        Option("min_cycles", default=3.0),
        Option("detrend", default=None),
        Option("detrend_window_hours", default=None),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    metric = str(ctx.option("metrics"))
    bins = int(ctx.option("bins"))
    period_bins = ctx.option("period_bins")
    raster_lut = ctx.option("trace_luts")
    hour_ticks = ctx.option("hour_ticks")
    order_requested = list(ctx.option("order"))
    secondary_requested = ctx.option("secondary_significance_method")
    panels = ctx.panels()

    frame = ctx.table("cell_frame.csv")
    if metric not in frame:
        numeric = [
            column for column in frame.select_dtypes(include=np.number).columns
            if column not in {"identity", "frame_index", "imagej_frame", "hours"}
        ]
        raise SystemExit(
            f"--metrics {metric} is absent from cell_frame.csv. Numeric measurements "
            f"available: {', '.join(numeric)}"
        )
    if not pd.api.types.is_numeric_dtype(frame[metric]):
        raise SystemExit(f"--metrics {metric} must be numeric")

    inherited = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    try:
        resolved = workbench.resolve_analysis_options(inherited, ctx.option)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    params = resolved["params"]
    search = [resolved["period_min_hours"], resolved["period_max_hours"]]
    period_edges = _period_edges(search, bins, period_bins)
    first_eight_hour_boundary = np.ceil(search[0] / 8.0) * 8.0
    period_peak_edges = np.unique(np.concatenate((
        [search[0]],
        np.arange(first_eight_hour_boundary, search[1], 8.0),
        [search[1]],
    )))
    fits = _fit_cells(frame, metric, resolved, secondary_requested)
    if fits.empty:
        raise SystemExit("no cell traces remain for rhythm testing")
    fits["analysis_parameters_json"] = json.dumps(
        params, sort_keys=True, default=str,
    )

    primary_method = resolved["significance_method"]
    secondary_method = (
        str(fits["secondary_significance_method"].dropna().iloc[0])
        if fits["secondary_significance_method"].notna().any() else None
    )
    estimator_method = resolved["method"]
    primary_label = workbench.PERIOD_METHODS[primary_method]["label"]
    secondary_label = (
        workbench.PERIOD_METHODS[secondary_method]["label"]
        if secondary_method else None
    )
    estimator_label = workbench.PERIOD_METHODS[estimator_method]["label"]

    significant = fits["significant"].fillna(False).astype(bool)
    period_available = fits["period_available"].fillna(False).astype(bool)
    histogram_mask = significant & period_available
    phase_available = pd.to_numeric(
        fits["phase_hours"], errors="coerce",
    ).notna()
    period_peak_mask = histogram_mask & phase_available
    fits["rhythm_rank"] = np.select(
        [significant, fits["test_status"].eq("ok")], [0, 1], default=2,
    )

    traces = rhythm_panels.detrended_traces(
        frame, fits["identity"], metric,
        method=resolved["detrend"],
        window_hours=resolved["detrend_window_hours"],
        detrend_options=params,
    )
    unordered_matrix = traces.pivot(
        index="identity", columns="hours", values="detrended_z",
    )
    resolved_order = _resolved_order(order_requested)
    trace_comparable = matrix_ordering.trace_comparable(unordered_matrix)
    fits["trace_comparable"] = fits["identity"].map(
        trace_comparable
    ).eq(True)
    resolved_order_columns = {
        column.removeprefix("-") for column in resolved_order
    }
    uses_continuum = bool(
        {"pattern_rank", "spectral_rank"} & resolved_order_columns
    )
    sort_order = (
        resolved_order if uses_continuum
        else ["-trace_comparable", *resolved_order]
    )
    verdict_masks = (
        significant,
        ~significant & fits["test_status"].eq("ok"),
        ~fits["test_status"].eq("ok"),
    )
    rank_methods = {
        "pattern_rank": "principal_component_gradient",
        "spectral_rank": "spectral_continuum",
    }
    for rank_column, ordering_method in rank_methods.items():
        if rank_column not in resolved_order_columns:
            continue
        ranks: dict[object, int] = {}
        for verdict_mask in verdict_masks:
            identities = fits.loc[verdict_mask, "identity"].tolist()
            ordered = matrix_ordering.trace_pattern_order(
                unordered_matrix.reindex(identities), method=ordering_method,
            )
            ranks.update({
                identity: rank for rank, identity in enumerate(ordered)
            })
        fits[rank_column] = fits["identity"].map(ranks)
    if "displayed_onset_hours" in resolved_order_columns:
        displayed_onsets = matrix_ordering.displayed_onsets(unordered_matrix)
        fits["displayed_onset_hours"] = fits["identity"].map(displayed_onsets)

    row_order_method = _order_label(order_requested)
    fits["row_order_method"] = row_order_method
    fits["row_order_keys_json"] = json.dumps(sort_order)
    significant_ids = matrix_ordering.ordered_identities(
        fits[significant], sort_order,
    )
    nonsignificant_ids = matrix_ordering.ordered_identities(
        fits[~significant & fits["test_status"].eq("ok")], sort_order,
    )
    untested_ids = matrix_ordering.ordered_identities(
        fits[~fits["test_status"].eq("ok")], sort_order,
    )
    order = significant_ids + nonsignificant_ids + untested_ids

    label = describe(metric).label
    cells_tested = int(fits["test_status"].eq("ok").sum())
    significant_count = int(significant.sum())
    histogram_count = int(histogram_mask.sum())
    not_tested_count = int(len(fits) - cells_tested)
    significant_periods = pd.to_numeric(
        fits.loc[histogram_mask, "period_hours"], errors="coerce"
    )
    median_period = (
        float(significant_periods.median()) if histogram_count else np.nan
    )
    edge_count = int(
        (histogram_mask & fits["period_at_search_edge"].fillna(False)).sum()
    )
    underdetermined_count = int(
        (histogram_mask & fits["period_underdetermined"].fillna(True)).sum()
    )
    span_hours = float(ctx.summary["hours_covered"])
    provenance = provenance_note(ctx.summary)
    correction = resolved["multiple_testing"]
    correction_label = {
        "bh": "Benjamini-Hochberg",
        "bonferroni": "Bonferroni",
        "sidak": "Sidak",
    }.get(correction, correction)
    correction_text = (
        "unadjusted p" if correction == "none"
        else f"{correction_label}-adjusted q"
    )
    detrend_label = str(resolved["detrend"]).replace("_", " ")
    verdict_text = (
        f"{primary_label} and {secondary_label} must both have {correction_text} "
        f"< {resolved['rhythmic_alpha']:g}"
        if secondary_label else
        f"{primary_label} {correction_text} < {resolved['rhythmic_alpha']:g}"
    )
    whole_trace_ordered = uses_continuum
    onset_ordered = "displayed_onset_hours" in resolved_order_columns
    footnote = (
        f"{estimator_label} estimated period over {search[0]:g}-{search[1]:g} h after "
        f"{detrend_label} detrending; {verdict_text} defines significant. "
        f"Periods supported by fewer than {resolved['min_cycles']:g} observed cycles "
        "are flagged but not removed from the significant-period distribution."
        + (
            f"\n{row_order_method.capitalize()} ordering uses a centred five-sample "
            "smoothed copy; the displayed values and times remain unchanged. It does "
            "not compare phase or imply synchrony."
            if whole_trace_ordered else ""
        )
        + (
            "\nDisplayed onset is the first three-sample blue-to-red transition in "
            "absolute recording hours; it is not a shared phase."
            if onset_ordered else ""
        )
        + (
            "\nThe peak-position matrix divides each fitted peak by that cell's own "
            "detected period. It is descriptive and does not establish a shared phase, "
            "synchrony or a common tissue clock."
            if "period_peak_matrix" in panels else ""
        )
        + (f"\n{provenance}" if provenance else "")
    )

    matrix = unordered_matrix.reindex(order)
    hours_axis = matrix.columns.to_numpy(float)
    traces_table = (
        matrix.reset_index()
        .melt(id_vars="identity", var_name="hours", value_name="detrended_z")
        .dropna(subset=["detrended_z"])
    )
    traces_table["row"] = traces_table["identity"].map(
        {identity: row for row, identity in enumerate(order)}
    )
    trace_columns = [
        "identity", "period_hours", "p_value", "q_value", "significant",
        "rhythm_status", "period_underdetermined", "period_at_search_edge",
        "period_estimation_method", "primary_significance_method",
        "secondary_significance_method", "workbench_version",
        "row_order_method", "row_order_keys_json", "trace_comparable",
    ]
    for column in (
        "pattern_rank", "spectral_rank", "displayed_onset_hours",
    ):
        if column in fits:
            trace_columns.append(column)
    traces_table = traces_table.merge(
        fits[trace_columns], on="identity", how="left",
    )

    width_in = 15.5
    header_in = 1.95
    gap_in = 2.05
    foot_lines = footnote.count("\n") + 1
    foot_in = 2.15 + 1.45 * ctx.theme.size("note") / 72.0 * max(0, foot_lines - 2)
    heights = {
        panel.key: panel.min_height_inches for panel in ctx.spec.panels
        if panel.key in panels
    }
    height_in = header_in + foot_in + sum(heights.values()) + gap_in * (len(panels) - 1)
    figure_ = ctx.sheet(width_in, height_in)

    def fraction(inches: float) -> float:
        return inches / height_in

    left = 0.085
    tops: dict[str, float] = {}
    cursor = height_in - header_in
    for name in panels.keys:
        tops[name] = cursor
        cursor -= heights[name] + gap_in

    axes = []
    auxiliary: dict[str, pd.DataFrame] = {
        "rhythm_fits.csv": fits,
        "statistics.csv": _statistics_table(fits),
    }
    figure_data = None
    lead = panels.keys[0]

    if "raster" in panels:
        bottom = fraction(tops["raster"] - heights["raster"])
        height = fraction(heights["raster"])
        ax_map = figure_.add_axes([left, bottom, 0.745, height])
        ax_strip = figure_.add_axes([0.845, bottom, 0.017, height])
        blocks = [
            (len(significant_ids), "rhythmic", "significant"),
            (len(nonsignificant_ids), "arrhythmic", "not significant"),
            (len(untested_ids), "missing", "not tested"),
        ]
        raster = ctx.drew("raster", rhythm_panels.trace_raster(
            ax_map, matrix.to_numpy(float), hours_axis, ctx.theme,
            blocks=blocks,
            cmap=resolve_look(ctx.theme, raster_lut).cmap if raster_lut else None,
            hour_ticks=hour_ticks,
        ))
        category_strip(
            ax_strip,
            [(count, role) for count, role, _ in blocks],
            ctx.theme,
        )
        colour_bar(
            figure_, raster.extra["handle"],
            [0.910, bottom + height * 0.17, 0.015, height * 0.64], ctx.theme,
            label=f"Detrended {label.lower()} (SD)",
        )
        axes.append(ax_map)
        if lead == "raster":
            figure_data = traces_table
        else:
            auxiliary["detrended_traces.csv"] = traces_table

    if "period_peak_matrix" in panels:
        bottom = fraction(
            tops["period_peak_matrix"] - heights["period_peak_matrix"]
        )
        height = fraction(heights["period_peak_matrix"])
        ax_period_peak = figure_.add_axes([left, bottom, 0.67, height])
        period_peak = ctx.drew(
            "period_peak_matrix",
            rhythm_panels.timing_by_period(
                ax_period_peak,
                timing_hours=fits["phase_hours"],
                cycle_hours=fits["period_hours"],
                detected_period_hours=fits["period_hours"],
                theme=ctx.theme,
                valid=histogram_mask,
                bins=bins,
                period_edges=period_peak_edges,
                timing_label=f"{estimator_label} peak",
                x_label="Peak position in each cell's detected cycle (%)",
            ),
        )
        colour_bar(
            figure_, period_peak.extra["handle"],
            [0.775, bottom + height * 0.17, 0.015, height * 0.64], ctx.theme,
            label="Significant cells per bin",
        )
        if lead == "period_peak_matrix":
            figure_data = period_peak.data
        else:
            auxiliary["period_peak_matrix.csv"] = period_peak.data
        axes.append(ax_period_peak)

    if "period_histogram" in panels:
        bottom = fraction(tops["period_histogram"] - heights["period_histogram"])
        height = fraction(heights["period_histogram"])
        ax_histogram = figure_.add_axes([left, bottom, 0.67, height])
        histogram = ctx.drew(
            "period_histogram",
            rhythm_panels.significant_period_histogram(
                ax_histogram, significant_periods, ctx.theme, bins=period_edges,
            ),
        )
        ax_histogram.set_xlim(period_edges[0], period_edges[-1])
        if lead == "period_histogram":
            figure_data = histogram.data
        else:
            auxiliary["significant_period_histogram.csv"] = histogram.data
        axes.append(ax_histogram)

    auxiliary.update(ctx.provenance_auxiliary())
    note_y = fraction(tops.get(
        "period_peak_matrix",
        tops.get("period_histogram", tops.get("raster", height_in)),
    ))
    median_text = f"{median_period:.2f} h" if np.isfinite(median_period) else "not available"
    verdict_rule_text = "dual-test" if secondary_label else "single-test"

    return FigureResult(
        figure=figure_,
        axes=axes,
        figure_data=figure_data,
        auxiliary=auxiliary,
        header_x=left,
        heading=f"{label} per cell",
        title_fields={
            "metric": label,
            "search": f"{search[0]:g}\u2013{search[1]:g}",
        },
        subtitle=(
            f"{ctx.summary['stem']}, {span_hours:.0f} h at "
            f"{ctx.summary['minutes_per_frame']:.0f} min per frame. "
            f"{cells_tested} cells tested; {significant_count} had a significant rhythm "
            f"under the selected {verdict_rule_text} rule.\nRows use {detrend_label} detrending "
            f"and {row_order_method} order within each verdict block."
        ),
        note=(
            f"Significant rhythms  {significant_count} / {cells_tested}\n"
            f"Peaks in matrix  {int(period_peak_mask.sum())}\n"
            f"Periods in histogram  {histogram_count}\n"
            f"Median period  {median_text}\n"
            f"At search boundary  {edge_count}\n"
            f"Fewer than {resolved['min_cycles']:g} cycles  {underdetermined_count}\n"
            f"Not tested  {not_tested_count}"
        ),
        note_at={
            "x": 0.835,
            "y": note_y,
            "fontsize": ctx.theme.size("subtitle"),
            "linespacing": 1.35,
        },
        footnote=footnote,
        readme=f"""## What the figure shows

**Ordered trace raster.** One row per cell shows {label.lower()} after
{detrend_label} detrending and within-cell scaling. Cells with a
significant rhythm are separated from completed non-significant tests and
traces that could not be tested.

By default, cells are sorted along a principal-component gradient: one
data-derived axis that summarises the dominant whole-trace blue/red progression.
The ordering calculation uses a centred five-sample smoothed copy, while the
raster retains the original detrended values and absolute recording times.
It is display ordering, not a phase comparison or evidence of synchrony or a
shared period.

The approved alternatives are `--order spectral` for a smooth continuum of
pairwise trace similarity, `--order onset,period` for the first sustained
blue-to-red transition in absolute recording hours followed by estimated
period, and `--order period` for the original estimated-period ordering.

**Period-by-peak-position matrix.** This panel uses only significant cells with
available {estimator_label} period and phase estimates. Rows group independently
estimated periods; columns show each fitted peak as a fraction of that same
cell's own detected cycle. It is a descriptive map, not a common phase axis or
evidence that cells are synchronized.

**Significant-period histogram.** Every bar counts only cells whose selected
rhythm verdict is significant and whose {estimator_label} period estimate is
available. No 24-hour folding or shared period is assumed. The matching table
also records each bin's fraction of the significant cells shown.

The default verdict uses {primary_label} alone and compares its raw p-value with
the selected alpha because p-value correction is off by default
(`--multiple-testing none`). Use `--multiple-testing bh`, `bonferroni` or
`sidak` to opt into correction across the tested cells. Setting
`--secondary-significance-method` opts into requiring both tests under the same
selected correction setting. `--fit-method` selects any estimator in Circadian
Workbench's live period-method registry; `--significance-method`, the complete
detrending controls, `--period-min-hours`, `--period-max-hours`,
`--rhythmic-alpha`, `--multiple-testing`, `--min-observations`, `--min-cycles`
and `--period-config` follow the shared circadian figure contract.

Each panel can be the whole page: `--panels raster`,
`--panels period_peak_matrix` or `--panels period_histogram`.

## What to be careful of

Statistical evidence, the estimated period and whether the recording contains
enough cycles to constrain that period remain separate columns. A significant
cell with fewer than {resolved['min_cycles']:g} observed cycles stays in the
histogram but is flagged in `rhythm_fits.csv`; a significant cell whose selected
estimator returned no period is counted in the note but cannot appear in a
period bin.""" + (f"\n\n{provenance}" if provenance else ""),
        console=(
            f"panels {panels.keys}  metric {metric}  cells {len(fits)}  "
            f"tested {cells_tested}  significant {significant_count}  "
            f"periods plotted {histogram_count}"
        ),
    )


if __name__ == "__main__":
    run_figure("cd68-reporter-rhythm", DEFAULT_RUN)
