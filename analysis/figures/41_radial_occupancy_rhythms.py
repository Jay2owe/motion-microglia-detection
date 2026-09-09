"""Figure 41: one broad-period radial-occupancy rhythm test per cell.

    python analysis/figures/41_radial_occupancy_rhythms.py <run>
    ... --period-min-hours 2 --period-max-hours 48
    ... --detrend linear --fit-method lomb
    ... --multiple-testing bh --rhythmic-alpha 0.05

Each cell-frame is reduced to the mean occupied fraction across equal bands of
relative soma distance. This is the normalised area under the radial occupancy
profile: a larger value means that more of the profile is yellow. It preserves
the coordinated wave in the Sholl kymograph without treating every band as a
separate hypothesis.
"""
from __future__ import annotations

import sys
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
from _options import commas
from _schema import FigureContext, FigureResult, Option, Panel, Table, figure, run_figure
from panels import common
from panels import rhythms as rhythm_panels


VALUE_COLUMN = "mean_relative_radial_occupancy"
ORDER_ALIASES = {
    "status": "rhythm_rank",
    "period": "period_hours",
    "significance": "q_value",
    "occupancy": "mean_occupancy",
    "identity": "identity",
}


def _resolved_order(requested: list[str]) -> list[str]:
    return matrix_ordering.resolve_order(requested, aliases=ORDER_ALIASES)


def _relative_occupancy_trace(
    sholl: pd.DataFrame,
    *,
    support_threshold: float,
    length_per_pixel: float,
) -> tuple[pd.DataFrame, int]:
    """One normalised profile area per cell-frame.

    All retained rings must have enough observable annulus in that frame. This
    keeps a ring disappearing at the image boundary from masquerading as an
    occupancy change.
    """
    ring_count = int(sholl["ring"].nunique())
    if ring_count < 1:
        raise SystemExit("no radial rings remain after filtering")

    inner_px = sholl["radius_inner"].to_numpy(float) / float(length_per_pixel)
    outer_px = sholl["radius_outer"].to_numpy(float) / float(length_per_pixel)
    expected = np.pi * (outer_px ** 2 - inner_px ** 2)
    prepared = sholl.copy()
    prepared["annulus_support_fraction"] = np.minimum(
        prepared["annulus_px"].to_numpy(float) / np.maximum(expected, 1e-12),
        1.0,
    )
    prepared["ring_usable"] = (
        prepared["annulus_support_fraction"].ge(float(support_threshold))
        & prepared["occupancy"].notna()
    )

    def reduce_frame(group: pd.DataFrame) -> pd.Series:
        usable = group["ring_usable"].to_numpy(bool)
        complete = len(group) == ring_count and int(usable.sum()) == ring_count
        return pd.Series({
            "hours": float(group["hours"].iloc[0]),
            "valid_rings": int(usable.sum()),
            "total_rings": ring_count,
            VALUE_COLUMN: (
                float(group.loc[usable, "occupancy"].mean()) if complete else np.nan
            ),
        })

    traces = (
        prepared.groupby(["identity", "frame_index"], sort=True, as_index=False)
        .apply(reduce_frame, include_groups=False)
        .reset_index(drop=True)
    )
    return traces, ring_count


def _add_display_values(
    traces: pd.DataFrame,
    *,
    display: str,
    detrend: str,
    window_hours: float,
    detrend_options: dict | None = None,
) -> pd.DataFrame:
    result = traces.copy()
    if display == "raw":
        result["display_value"] = result[VALUE_COLUMN]
        return result
    if display != "detrended":
        raise SystemExit("--display must be raw or detrended")

    result["display_value"] = np.nan
    for _, group in result.groupby("identity", sort=True):
        usable = group[["hours", VALUE_COLUMN]].dropna().sort_values("hours")
        if len(usable) < 2:
            continue
        result.loc[usable.index, "display_value"] = rhythm_panels.detrended_z(
            usable["hours"], usable[VALUE_COLUMN],
            method=detrend, window_hours=window_hours,
            detrend_options=detrend_options,
        )
    return result


@figure(
    number=41,
    slug="radial-occupancy-rhythms",
    summary="one broad-period test of integrated relative radial occupancy per cell",
    title="A single radial-occupancy rhythm trace for every cell",
    grammar="cell-by-time matrix with a family-corrected rhythm verdict and estimated period",
    reads=(Table("sholl.csv", module="sholl"),),
    panels=(
        Panel(
            "matrix", common.raster,
            title=(
                "Each row is one cell; colour is the mean occupied fraction "
                "across its relative-distance bands"
            ),
            min_width_inches=10.0, min_height_inches=12.0,
        ),
    ),
    options=(
        Option("rings", default=None),
        Option("scaling", default="cell"),
        Option("display", default="raw", metavar="raw|detrended",
               help="show the measured 0-1 occupancy or its detrended within-cell change"),
        Option("fit_method", default=None),
        Option("significance_method", default=None),
        Option("period_config", default={}),
        Option("detrend", default=None),
        Option("detrend_window_hours", default=None),
        Option("period_min_hours", default=2.0),
        Option("period_max_hours", default=48.0),
        Option("rhythmic_alpha", default=0.05),
        Option("multiple_testing", default="bh"),
        Option("min_observations", default=24),
        Option("min_cycles", default=3.0),
        Option("annulus_support", default=0.5),
        Option("order", default=["status", "period"], cast=commas,
               metavar="COL,COL",
               help="cell-row ordering; aliases are status, period, significance, occupancy and identity"),
        Option("matrix_lut", default=None),
        Option("hour_ticks", default=None),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    sholl = ctx.table("sholl.csv")
    if "scaling" not in sholl:
        raise SystemExit(
            "this sholl.csv predates labelled radial scalings; re-run the sholl module"
        )
    scaling = str(ctx.option("scaling"))
    if scaling not in {"cell", "global"}:
        raise SystemExit("--scaling must be cell or global")
    sholl = sholl[sholl["scaling"].eq(scaling)].copy()
    if sholl.empty:
        raise SystemExit(f"this run has no {scaling!r} radial profile")

    requested_rings = ctx.option("rings")
    if requested_rings is not None:
        if int(requested_rings) < 1:
            raise SystemExit("--rings needs a positive number of inner rings")
        sholl = sholl[sholl["ring"] < int(requested_rings)].copy()

    support_threshold = float(ctx.option("annulus_support"))
    if not 0.0 < support_threshold <= 1.0:
        raise SystemExit("--annulus-support must be above 0 and at most 1")
    traces, ring_count = _relative_occupancy_trace(
        sholl,
        support_threshold=support_threshold,
        length_per_pixel=(
            float(ctx.scale.microns_per_pixel) if ctx.scale.calibrated else 1.0
        ),
    )

    inherited = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    resolved = workbench.resolve_analysis_options(inherited, ctx.option)
    params = resolved["params"]
    period_min = resolved["period_min_hours"]
    period_max = resolved["period_max_hours"]
    alpha = resolved["rhythmic_alpha"]
    min_cycles = resolved["min_cycles"]
    min_observations = resolved["min_observations"]
    detrend = resolved["detrend"]
    method = resolved["method"]
    significance_method = resolved["significance_method"]
    fits = workbench.estimate_grouped_rhythms(
        traces,
        group_columns=["identity"],
        value_column=VALUE_COLUMN,
        params=params,
        method=method,
        significance_method=significance_method,
        detrend=detrend,
        detrend_window_hours=resolved["detrend_window_hours"],
        min_observations=min_observations,
        correction=resolved["multiple_testing"],
        min_cycles=min_cycles,
    )
    if fits.empty:
        raise SystemExit("no cell traces remain after radial-profile reduction")

    observed = traces.groupby("identity", sort=True).agg(
        observed_frames=(VALUE_COLUMN, "count"),
        mean_occupancy=(VALUE_COLUMN, "mean"),
    ).reset_index()
    cell_summary = fits.merge(observed, on="identity", how="left", validate="one_to_one")
    underdetermined = cell_summary.get(
        "period_underdetermined", pd.Series(True, index=cell_summary.index)
    ).astype("boolean").fillna(True).astype(bool)
    cell_summary["rhythm_rank"] = np.select(
        [
            cell_summary["rhythm_status"].eq("rhythmic") & ~underdetermined,
            cell_summary["rhythm_status"].eq("rhythmic"),
            cell_summary["rhythm_status"].eq("not rhythmic"),
        ],
        [0, 1, 2],
        default=3,
    )
    order_requested = list(ctx.option("order"))
    order = matrix_ordering.ordered_identities(
        cell_summary, _resolved_order(order_requested)
    )
    cell_summary["display_row"] = cell_summary["identity"].map(
        {identity: row + 1 for row, identity in enumerate(order)}
    )

    display = str(ctx.option("display"))
    traces = _add_display_values(
        traces,
        display=display,
        detrend=detrend,
        window_hours=float(params["detrend_window_hours"]),
        detrend_options=params,
    )
    hours = np.sort(traces["hours"].dropna().unique().astype(float))
    trace_matrix = traces.pivot(
        index="identity", columns="hours", values="display_value"
    ).reindex(index=order, columns=hours)

    status_rows = cell_summary.set_index("identity").reindex(order)
    periods = status_rows["period_hours"].where(
        status_rows["rhythm_status"].eq("rhythmic")
    ).to_numpy(float)[:, None]
    statuses = status_rows["rhythm_status"].fillna("not tested").to_numpy(object)[:, None]
    unavailable = status_rows["rhythm_status"].eq("rhythmic") & ~status_rows["period_available"].fillna(False)
    statuses[unavailable.to_numpy(), 0] = "period unavailable"
    uncertain = status_rows.get(
        "period_underdetermined", pd.Series(False, index=status_rows.index)
    ).astype("boolean").fillna(False).to_numpy(dtype=bool)[:, None]

    height_inches = 4.8 + 0.18 * len(order)
    figure_ = ctx.sheet(16.0, height_inches)
    bottom_inches = 1.95
    top_inches = 1.75
    drawing_height = (height_inches - bottom_inches - top_inches) / height_inches
    drawing_bottom = bottom_inches / height_inches
    ax = figure_.add_axes([0.12, drawing_bottom, 0.56, drawing_height])
    status_ax = figure_.add_axes([0.77, drawing_bottom, 0.055, drawing_height])
    ax.set_title(
        ctx.spec.panel("matrix").heading(), loc="left",
        fontsize=ctx.theme.size("panel"), fontweight="bold",
    )

    if display == "raw":
        value_min, value_max = 0.0, 1.0
        value_cmap = ctx.option("matrix_lut")
        value_label = "Mean radial occupancy (fraction)"
        colour_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    else:
        matrix_values = trace_matrix.to_numpy(float)
        finite = np.abs(matrix_values[np.isfinite(matrix_values)])
        limit = float(np.percentile(finite, 98)) if finite.size else 1.0
        limit = max(limit, 1e-9)
        value_min, value_max = -limit, limit
        value_cmap = (
            ctx.option("matrix_lut")
            if ctx.option("matrix_lut") is not None
            else ctx.theme["diverging_cmap"]
        )
        value_label = "Detrended occupancy change (within-cell standard deviations)"
        colour_ticks = [-limit, 0.0, limit]

    raster = ctx.drew("matrix", common.raster(
        ax, trace_matrix.to_numpy(float), ctx.theme,
        hours=hours, cmap=value_cmap, vmin=value_min, vmax=value_max,
    ))
    ax.set_xlabel("Hours from start of recording")
    hour_step = ctx.option("hour_ticks")
    ax.set_xticks(ctx.theme.hour_ticks(
        float(hours.min()), float(hours.max()),
        None if hour_step is None else float(hour_step),
    ))
    ax.set_ylabel("Cell identity")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(identity) for identity in order])
    ax.tick_params(axis="y", labelsize=max(6.0, ctx.theme.size("caption") * 0.62))
    common.inset_colour_bar(
        ax, raster.extra["handle"], ctx.theme,
        label=value_label, ticks=colour_ticks,
        height_inches=5.0, width_inches=0.18,
    )

    status = rhythm_panels.period_status_matrix(
        status_ax,
        periods, statuses, ctx.theme,
        row_labels=[str(identity) for identity in order],
        column_labels=["Rhythm"],
        underdetermined=uncertain,
        period_min=period_min, period_max=period_max,
        cmap=None,
        x_label="", y_label="",
    )
    status_ax.set_yticks([])
    status_ax.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False,
    )
    status_ax.set_xticklabels(["Best\nperiod"], rotation=0)
    period_ticks = np.unique(np.array(
        [period_min, 8, 16, 24, 32, 40, period_max], dtype=float
    ))
    period_ticks = period_ticks[
        (period_ticks >= period_min) & (period_ticks <= period_max)
    ]
    period_bar = common.inset_colour_bar(
        status_ax, status.extra["handle"], ctx.theme,
        label="Estimated period in rhythmic traces (h)", ticks=period_ticks,
        height_inches=5.0, width_inches=0.18,
    )
    period_bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ctx.theme.legend(
        status_ax, status.extra["legend_handles"],
        [handle.get_label() for handle in status.extra["legend_handles"]],
        loc="upper left", bbox_to_anchor=(2.8, 1.0), ncol=1,
    )
    status_ax._semantic_legend_handled = True

    tested = int(fits["test_status"].eq("ok").sum())
    rhythmic = int(fits["significant"].sum())
    underdetermined_count = int(
        (fits["significant"] & fits["period_underdetermined"]).sum()
    )
    span = float(ctx.summary["hours_covered"])
    resolvable_period = span / min_cycles
    method_label = workbench.PERIOD_METHODS[method]["label"]
    test_label = workbench.PERIOD_METHODS[significance_method]["label"]
    correction_label = {
        "bh": "Benjamini–Hochberg false-discovery rate",
        "bonferroni": "Bonferroni",
        "sidak": "Šidák",
        "none": "no multiple-testing correction",
    }.get(resolved["multiple_testing"], resolved["multiple_testing"])
    scaling_text = (
        "bands dividing each cell's current 95% reach"
        if scaling == "cell" else "fixed-width bands across the movie"
    )
    footnote = (
        f"Each value is the normalised area under {ring_count} radial occupancy bands: "
        f"their mean occupied fraction, using {scaling_text}. {method_label} searched "
        f"{period_min:g}–{period_max:g} h after {detrend} detrending; "
        f"{test_label} supplied the rhythm-test p-values; {correction_label}, "
        f"q < {alpha:g}, across {tested} tested cells. This tests the trace, not an "
        f"individual fitted component. Hatching marks a rhythmic trace with no eligible period fit. Black outlines "
        f"mark candidates with fewer than {min_cycles:g} cycles ({resolvable_period:.1f} h "
        f"in this {span:.1f} h record), so those period estimates are exploratory. A cell-frame "
        f"is omitted unless every band has at least {support_threshold:.0%} of its annulus available."
    )

    statistics_columns = [
        "identity", "metric", "method", "significance_method", "observations", "span_hours",
        "estimate_status", "estimate_reason", "period_available", "estimator_p_value",
        "significance_period_hours", "period_difference_hours", "period_error_hours",
        "phase_hours", "phase_error_hours", "amplitude", "amplitude_error", "rae", "goodness_of_fit",
        "period_hours", "p_value", "q_value", "correction", "alpha",
        "significant", "rhythm_status", "test_status", "reason",
        "periodogram_significant", "peak_power", "threshold", "detrend",
        "detrend_window_hours",
        "cycles_observed", "period_underdetermined", "period_at_search_edge",
        "period_search_min_hours", "period_search_max_hours", "family_tests",
        "workbench_version",
    ]
    statistics = fits[[column for column in statistics_columns if column in fits]].copy()
    figure_data = traces.merge(
        cell_summary[[
            "identity", "display_row", "rhythm_status", "period_hours",
            "q_value", "significant", "period_underdetermined",
        ]],
        on="identity", how="left", validate="many_to_one",
    )
    auxiliary = {
        "statistics.csv": statistics,
        "cell_order.csv": cell_summary,
        **ctx.provenance_auxiliary(),
    }
    return FigureResult(
        figure=figure_,
        axes=[ax, status_ax],
        figure_data=figure_data,
        subtitle=(
            f"{len(order)} cells; {rhythmic} of {tested} testable cells were rhythmic "
            f"after correction. Rows ordered by {', then '.join(order_requested)}."
        ),
        footnote=footnote,
        auxiliary=auxiliary,
        header_x=0.12,
        readme=f"""## What the figure shows

Each row is one cell through time. The main matrix shows one value per frame:
the mean occupied fraction across {ring_count} equal bands of relative distance
from the soma. This is the normalised area under the radial occupancy profile,
so a yellow wave spanning several bands becomes a rising single trace rather
than {ring_count} separately tested hypotheses.

The right strip reports one broad-period rhythm test per cell. Colour gives the
estimated period in a rhythmic trace; grey is a completed non-significant test and pale
grey is a cell that could not be tested. A black outline means fewer than
{min_cycles:g} cycles of that period fit into the observed duration.

## Analysis

{method_label} was run through Circadian Workbench {workbench.WORKBENCH_VERSION}
after {detrend} detrending. The search covered {period_min:g}–{period_max:g} hours
and did not assume a 24-hour period. Raw false-alarm probabilities were adjusted
across cells using {correction_label}; q < {alpha:g} defined rhythmic cells.

The plotted value uses {scaling_text}. A frame is excluded unless every retained
ring has at least {support_threshold:.0%} of its full geometric annulus available,
preventing image-boundary loss from looking like a biological occupancy change.

`sholl_auc` is not used here: that existing column is the area under the radial
crossing-count profile, not the area under occupancy.
""",
        console=(
            f"{rhythmic} corrected rhythmic cell results; "
            f"{underdetermined_count} cover fewer than {min_cycles:g} cycles"
        ),
    )


if __name__ == "__main__":
    run_figure("radial-occupancy-rhythms")
