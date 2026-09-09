"""Figure 42: broad-period rhythm results for user-selected cell metrics.

    python analysis/figures/42_metric_rhythm_matrix.py <run>
    ... --metrics corrected_mean,area_px,speed,reach_p95
    ... --fit-method lomb --period-min-hours 2 --period-max-hours 48
    ... --detrend linear --multiple-testing bh
    ... --correction-scope matrix --rhythmic-alpha 0.05

Rows are cells and columns are arbitrary documented measurements from
``cell_frame.csv``. Colour is the estimated period in a rhythmic trace. The figure
does not contain a metric-specific calculation and does not assume 24 hours.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
import matrix_ordering
from _metrics import semantic_label
from _options import commas
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from panels import common
from panels import rhythms as rhythm_panels


ORDER_ALIASES = {
    "period": "median_rhythmic_period_hours",
    "significance": "best_q_value",
    "rhythmic_fraction": "rhythmic_fraction",
    "rhythmic_count": "rhythmic_metrics",
    "identity": "identity",
}

METHOD_LABELS = {
    "lomb": "Lomb–Scargle periodogram",
    "chi_square": "Enright chi-square periodogram",
    "f": "F periodogram",
    "jtk": "JTK_CYCLE",
    "ejtk": "empirical JTK_CYCLE",
}


def _resolved_order(requested: list[str]) -> list[str]:
    return matrix_ordering.resolve_order(requested, aliases=ORDER_ALIASES)


def _test_metrics(
    long: pd.DataFrame,
    *,
    metrics: list[str],
    params: dict,
    method: str,
    detrend: str,
    detrend_window_hours: float,
    min_observations: int,
    correction: str,
    correction_scope: str,
    min_cycles: float,
    significance_method: str | None = None,
) -> pd.DataFrame:
    """Test one trace per cell and metric under the requested correction family."""
    common_args = {
        "value_column": "value",
        "params": params,
        "method": method,
        "significance_method": significance_method,
        "detrend": detrend,
        "detrend_window_hours": detrend_window_hours,
        "min_observations": min_observations,
        "correction": correction,
        "min_cycles": min_cycles,
    }
    if correction_scope == "matrix":
        fits = workbench.estimate_grouped_rhythms(
            long,
            group_columns=["identity", "measurement"],
            **common_args,
        )
        fits = fits.drop(columns="metric").rename(columns={"measurement": "metric"})
        fits["correction_family"] = "all selected cells and metrics"
        return fits
    if correction_scope != "metric":
        raise SystemExit("--correction-scope must be matrix or metric")

    pieces = []
    for metric in metrics:
        selected = long[long["measurement"].eq(metric)].copy()
        fit = workbench.estimate_grouped_rhythms(
            selected,
            group_columns=["identity"],
            **common_args,
        )
        fit["metric"] = metric
        fit["correction_family"] = f"cells within {metric}"
        pieces.append(fit)
    return pd.concat(pieces, ignore_index=True)


@figure(
    number=42,
    slug="metric-rhythm-matrix",
    summary="broad-period rhythm status and estimated period for arbitrary cell metrics",
    title="Broad-period rhythms across selected cell measurements",
    grammar="cell-by-metric matrix of corrected rhythm status and estimated period",
    reads=(Table("cell_frame.csv", module="measurement modules"),),
    panels=(
        Panel(
            "matrix", rhythm_panels.period_status_matrix,
            title="Each row is a cell; each column is a user-selected measurement",
            min_width_inches=10.0, min_height_inches=12.0,
        ),
    ),
    options=(
        Option(
            "metrics",
            default=["corrected_mean", "area_px", "speed", "reach_p95"],
            help="documented numeric cell-frame measurements, one matrix column each",
        ),
        Option("fit_method", default=None),
        Option("significance_method", default=None),
        Option("period_config", default={}),
        Option("detrend", default=None),
        Option("detrend_window_hours", default=None),
        Option("period_min_hours", default=2.0),
        Option("period_max_hours", default=48.0),
        Option("rhythmic_alpha", default=0.05),
        Option("multiple_testing", default="bh"),
        Option("correction_scope", default="matrix"),
        Option("min_observations", default=24),
        Option("min_cycles", default=3.0),
        Option(
            "order", default=["-rhythmic_fraction", "period"], cast=commas,
            metavar="COL,COL",
            help=(
                "cell-row ordering; aliases are rhythmic_fraction, rhythmic_count, "
                "period, significance and identity"
            ),
        ),
        Option("matrix_lut", default=None),
        Option("column_label_rotation", default=0.0),
        Option("column_label_wrap", default=18),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    frame = ctx.table("cell_frame.csv")
    metrics = [str(metric) for metric in ctx.option("metrics")]
    if not metrics:
        raise SystemExit("--metrics needs at least one cell-frame measurement")
    if len(set(metrics)) != len(metrics):
        raise SystemExit("--metrics contains the same measurement more than once")
    unknown = [metric for metric in metrics if metric not in frame]
    if unknown:
        numeric = [
            column for column in frame.select_dtypes(include=np.number).columns
            if column not in {"identity", "frame_index", "imagej_frame", "hours"}
        ]
        raise SystemExit(
            f"--metrics names {', '.join(unknown)}, which cell_frame.csv does not contain. "
            f"Numeric measurements available: {', '.join(numeric)}"
        )
    nonnumeric = [metric for metric in metrics if not pd.api.types.is_numeric_dtype(frame[metric])]
    if nonnumeric:
        raise SystemExit("--metrics must be numeric: " + ", ".join(nonnumeric))
    try:
        labels = [semantic_label(metric) for metric in metrics]
    except ValueError as error:
        raise SystemExit(str(error)) from None
    label_wrap = int(ctx.option("column_label_wrap"))
    if label_wrap < 0:
        raise SystemExit("--column-label-wrap must be 0 or a positive number")
    if label_wrap:
        labels = [
            textwrap.fill(
                label, width=max(8, label_wrap),
                break_long_words=False, break_on_hyphens=False,
            )
            for label in labels
        ]

    inherited = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    resolved = workbench.resolve_analysis_options(inherited, ctx.option)
    params = resolved["params"]
    period_min = resolved["period_min_hours"]
    period_max = resolved["period_max_hours"]
    alpha = resolved["rhythmic_alpha"]
    min_cycles = resolved["min_cycles"]
    min_observations = resolved["min_observations"]
    method = resolved["method"]
    significance_method = resolved["significance_method"]
    detrend = resolved["detrend"]
    correction = resolved["multiple_testing"]
    correction_scope = str(ctx.option("correction_scope"))

    long = frame[["identity", "hours", *metrics]].melt(
        id_vars=["identity", "hours"],
        value_vars=metrics,
        var_name="measurement",
        value_name="value",
    )
    fits = _test_metrics(
        long,
        metrics=metrics,
        params=params,
        method=method,
        significance_method=significance_method,
        detrend=detrend,
        detrend_window_hours=resolved["detrend_window_hours"],
        min_observations=min_observations,
        correction=correction,
        correction_scope=correction_scope,
        min_cycles=min_cycles,
    )
    if fits.empty:
        raise SystemExit("none of the selected cell-metric traces could be assessed")

    summary = fits.groupby("identity", sort=True).agg(
        tested_metrics=("test_status", lambda values: int(values.eq("ok").sum())),
        rhythmic_metrics=("significant", "sum"),
        best_q_value=("q_value", "min"),
    ).reset_index()
    significant_periods = (
        fits[fits["significant"]]
        .groupby("identity")["period_hours"]
        .median()
        .rename("median_rhythmic_period_hours")
    )
    summary = summary.merge(significant_periods, on="identity", how="left")
    summary["rhythmic_fraction"] = np.divide(
        summary["rhythmic_metrics"], summary["tested_metrics"],
        out=np.zeros(len(summary), dtype=float),
        where=summary["tested_metrics"].to_numpy() > 0,
    )
    order_requested = list(ctx.option("order"))
    order = matrix_ordering.ordered_identities(
        summary, _resolved_order(order_requested)
    )
    summary["display_row"] = summary["identity"].map(
        {identity: row + 1 for row, identity in enumerate(order)}
    )

    matrix_index = pd.MultiIndex.from_product(
        [order, metrics], names=["identity", "metric"]
    )
    indexed = fits.set_index(["identity", "metric"]).reindex(matrix_index)
    periods = indexed["period_hours"].where(
        indexed["rhythm_status"].eq("rhythmic")
    ).to_numpy(float).reshape(len(order), len(metrics))
    statuses = (
        indexed["rhythm_status"].fillna("not tested")
        .to_numpy(object).reshape(len(order), len(metrics))
    )
    unavailable = indexed["rhythm_status"].eq("rhythmic") & ~indexed["period_available"].fillna(False)
    statuses[unavailable.to_numpy().reshape(statuses.shape)] = "period unavailable"
    uncertain = (
        indexed.get("period_underdetermined", pd.Series(False, index=indexed.index))
        .astype("boolean").fillna(False)
        .to_numpy(dtype=bool).reshape(len(order), len(metrics))
    )

    panel_width = max(10.0, 2.35 * len(metrics))
    height_inches = 4.8 + 0.18 * len(order)
    figure_width = panel_width + 4.5
    figure_ = ctx.sheet(figure_width, height_inches)
    bottom_inches = 1.95
    top_inches = 1.7
    left_inches = 1.9
    right_inches = 3.8
    ax = figure_.add_axes([
        left_inches / figure_width,
        bottom_inches / height_inches,
        panel_width / figure_width,
        (height_inches - bottom_inches - top_inches) / height_inches,
    ])
    ax.set_title(
        ctx.spec.panel("matrix").heading(), loc="left",
        fontsize=ctx.theme.size("panel"), fontweight="bold",
    )
    matrix = ctx.drew("matrix", rhythm_panels.period_status_matrix(
        ax,
        periods, statuses, ctx.theme,
        row_labels=[str(identity) for identity in order],
        column_labels=labels,
        underdetermined=uncertain,
        period_min=period_min, period_max=period_max,
        cmap=ctx.option("matrix_lut"),
        column_label_rotation=float(ctx.option("column_label_rotation")),
        x_label="Cell-frame measurement", y_label="Cell identity",
    ))
    colour_candidates = np.array(
        [period_min, 8, 16, 24, 32, 40, period_max], dtype=float
    )
    colour_ticks = np.unique(colour_candidates[
        (colour_candidates >= period_min) & (colour_candidates <= period_max)
    ])
    colour_bar = common.inset_colour_bar(
        ax, matrix.extra["handle"], ctx.theme,
        label="Estimated period in rhythmic traces (h)", ticks=colour_ticks,
        height_inches=5.0,
    )
    colour_bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ctx.theme.legend(
        ax, matrix.extra["legend_handles"],
        [handle.get_label() for handle in matrix.extra["legend_handles"]],
        loc="upper left", bbox_to_anchor=(1.14, 1.0), ncol=1,
    )
    ax._semantic_legend_handled = True

    tested = int(fits["test_status"].eq("ok").sum())
    rhythmic = int(fits["significant"].sum())
    rhythmic_cells = int(fits.loc[fits["significant"], "identity"].nunique())
    underdetermined_count = int(
        (fits["significant"] & fits["period_underdetermined"]).sum()
    )
    span = float(ctx.summary["hours_covered"])
    resolvable_period = span / min_cycles
    method_label = METHOD_LABELS.get(
        method, workbench.PERIOD_METHODS[method]["label"]
    )
    test_label = workbench.PERIOD_METHODS[significance_method]["label"]
    correction_label = {
        "bh": "Benjamini–Hochberg false-discovery rate",
        "bonferroni": "Bonferroni",
        "sidak": "Šidák",
        "none": "no multiple-testing correction",
    }.get(correction, correction)
    family_text = (
        "all selected cell–metric tests together"
        if correction_scope == "matrix"
        else "the cells within each metric separately"
    )
    footnote = (
        f"{method_label} searched {period_min:g}–{period_max:g} h after {detrend} "
        f"detrending. Rhythmicity: {test_label}; {correction_label}, q < {alpha:g}, across {family_text}. Black "
        f"outlines mark significant candidates with fewer than {min_cycles:g} cycles "
        f"({resolvable_period:.1f} h in this {span:.1f} h record), so those period "
        "estimates are exploratory. Grey is tested but not rhythmic; pale grey is not tested. "
        "Hatching marks rhythmic traces whose selected period fit failed. "
        "The test assesses the trace; its p-value is not a significance test of a separate fitted period."
    )
    fits["correction_scope"] = correction_scope
    fits["display_row"] = fits["identity"].map(
        {identity: row + 1 for row, identity in enumerate(order)}
    )
    fits["display_column"] = fits["metric"].map(
        {metric: column + 1 for column, metric in enumerate(metrics)}
    )
    statistics_columns = [
        "identity", "metric", "method", "significance_method", "observations", "span_hours",
        "estimate_status", "estimate_reason", "period_available", "estimator_p_value", "significance_period_hours",
        "period_difference_hours", "period_error_hours", "phase_hours", "phase_error_hours",
        "amplitude", "amplitude_error", "rae", "goodness_of_fit",
        "period_hours", "p_value", "q_value", "correction", "correction_scope",
        "correction_family", "alpha", "significant", "rhythm_status",
        "test_status", "reason", "periodogram_significant", "peak_power",
        "threshold", "detrend", "detrend_window_hours", "cycles_observed",
        "period_underdetermined",
        "period_at_search_edge", "period_search_min_hours",
        "period_search_max_hours", "family_tests", "workbench_version",
    ]
    statistics = fits[[column for column in statistics_columns if column in fits]].copy()
    auxiliary = {
        "statistics.csv": statistics,
        "cell_order.csv": summary,
        **ctx.provenance_auxiliary(),
    }
    return FigureResult(
        figure=figure_,
        axes=[ax],
        figure_data=fits,
        subtitle=(
            f"{rhythmic} of {tested} testable cell–metric traces were rhythmic after "
            f"correction, spanning {rhythmic_cells} of {len(order)} cells and "
            f"{len(metrics)} measurements."
        ),
        footnote=footnote,
        auxiliary=auxiliary,
        header_x=left_inches / figure_width,
        readme=f"""## What the figure shows

Rows are all {len(order)} cells and columns are the requested measurements from
`cell_frame.csv`: {', '.join(metrics)}. A coloured matrix cell passed the
corrected rhythm test; colour is its estimated period within the
{period_min:g}–{period_max:g}-hour search. Grey is a completed non-significant
test, pale grey is an unavailable test, and a black outline marks a significant
period represented by fewer than {min_cycles:g} observed cycles.

## Analysis

{method_label} estimated periods through Circadian Workbench {workbench.WORKBENCH_VERSION}
after {detrend} detrending. The period search did not assume a 24-hour cycle.
{test_label} supplied the raw probabilities, adjusted with {correction_label} across {family_text};
q < {alpha:g} defined rhythmicity. Exact raw and corrected values are in
`data/der/statistics.csv`.
The significance test concerns the trace, not an individual fitted component.
Hatching identifies a rhythmic trace whose period estimator could not supply a period.

## Reuse

`--metrics` accepts any documented numeric measurement in `cell_frame.csv` and
preserves the supplied column order. `--fit-method` accepts all Circadian Workbench
estimators, including `fft_nlls`. `--significance-method` separately selects a
statistical test (default Lomb-Scargle for estimators without one).
`--period-config` accepts method-specific settings as a JSON object.
Relative amplitude error (`rae`) is fit uncertainty, not a p-value.
The search limits, detrending, corrected alpha, correction method, correction
scope, minimum observations and minimum cycles are all independent options.
The reusable `panels.rhythms.period_status_matrix` panel contains no cell- or
measurement-specific calculation.
""",
        console=(
            f"{rhythmic} corrected rhythmic cell-metric results; "
            f"{underdetermined_count} cover fewer than {min_cycles:g} cycles"
        ),
    )


if __name__ == "__main__":
    run_figure("metric-rhythm-matrix")
