"""Figure 8: where every cell went over the whole recording.

One panel, ``panels.motility.trajectory_map``, which cuts each path at the gaps
rather than bridging them::

    python analysis/figures/08_identity_trajectories.py <run>
    ... --metrics total_path_px      colour the paths by a different track column
    ... --line-width-metric area_px_median
    ... --line-width-range 0.55,2.8
    ... --line-dash-metric rhythm_period_band
    ... --rhythm-metric corrected_mean --period-bins 2,12,20,28,48
    ... --trace-luts magma           the map that colour runs along

Colour and line width accept numeric per-cell columns from ``cell_summary.csv``.
Dash accepts a categorical column, low/medium/high thirds of a numeric column,
or ``rhythm_period_band``: significant periods for ``--rhythm-metric`` split
at ``--period-bins``, with separate not-rhythmic and unavailable categories.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis import circadian as workbench
from _metrics import describe
from _options import numbers
from _schema import (FigureContext, FigureResult, Option, Panel, Table, figure,
                     run_figure)
from panels import colour_bar, resolve_look
from panels import motility as motility_panels

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"

DEFAULT_PERIOD_BINS = [2.0, 12.0, 20.0, 28.0, 48.0]
DASH_PATTERNS = (
    "solid",
    (0, (10, 3)),
    (0, (3, 2)),
    (0, (10, 2, 2, 2)),
    (0, (1, 1.5)),
    (0, (6, 2, 1.5, 2, 1.5, 2)),
    (0, (12, 3, 3, 3)),
    (0, (5, 2, 5, 2, 1.5, 2)),
    (0, (2, 1.5, 8, 1.5)),
)
RHYTHM_METHOD_LABELS = {
    "lomb": "Lomb-Scargle periodogram",
    "chi_square": "Enright-Sokolove periodogram",
    "f": "F periodogram",
    "jtk": "JTK_CYCLE",
    "ejtk": "empirical JTK_CYCLE",
}


def _metric_values(tracks: pd.DataFrame, column: str, option: str) -> pd.Series:
    if column not in tracks:
        available = ", ".join(c for c in tracks.columns if tracks[c].dtype.kind in "bif")
        raise SystemExit(
            f"--{option.replace('_', '-')} {column} is not in cell_summary.csv. "
            f"Numeric columns available: {available}"
        )
    values = pd.to_numeric(tracks[column], errors="coerce")
    if not np.isfinite(values).any():
        raise SystemExit(
            f"--{option.replace('_', '-')} {column} has no finite numeric values"
        )
    return values


def _scaled_line_widths(
    values: pd.Series,
    base_width: float,
    requested_range: list[float],
) -> tuple[pd.Series, list[tuple[str, float, float]]]:
    """Map the central 90% of a numeric cell metric onto positive line widths."""
    if len(requested_range) != 2:
        raise SystemExit("--line-width-range needs exactly two comma-separated values")
    small, large = map(float, requested_range)
    if not (0 < small <= large and np.isfinite([small, large]).all()):
        raise SystemExit("--line-width-range values must be finite, positive, and ascending")

    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    lower, middle, upper = map(float, finite.quantile([0.05, 0.50, 0.95]))
    if upper > lower:
        fraction = ((numeric - lower) / (upper - lower)).clip(0.0, 1.0)
        widths = base_width * (small + fraction * (large - small))
        legend_values = [
            ("5th percentile", lower),
            ("Median", middle),
            ("95th percentile", upper),
        ]
    else:
        widths = pd.Series(base_width, index=numeric.index, dtype=float)
        legend_values = [("All cells", middle)]
    widths = widths.fillna(base_width)

    def width_for(value: float) -> float:
        if upper <= lower:
            return base_width
        fraction = np.clip((value - lower) / (upper - lower), 0.0, 1.0)
        return float(base_width * (small + fraction * (large - small)))

    legend = [(name, value, width_for(value)) for name, value in legend_values]
    return widths, legend


def _rhythm_period_categories(
    rhythms: pd.DataFrame,
    identities: pd.Index,
    rhythm_metric: str,
    period_bins: list[float],
) -> tuple[pd.Series, list[str], pd.DataFrame, str, str]:
    """Per-cell rhythm period bands, with estimator and verdict source named."""
    edges = np.asarray(period_bins, dtype=float)
    if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise SystemExit(
            "--period-bins needs at least two finite, ascending hour boundaries"
        )
    if "metric" not in rhythms or rhythm_metric not in set(rhythms["metric"].dropna()):
        available = ", ".join(sorted(map(str, set(rhythms.get("metric", [])))))
        raise SystemExit(
            f"--rhythm-metric {rhythm_metric} is not in rhythms.csv. Available: {available}"
        )

    fits = rhythms[rhythms["metric"] == rhythm_metric].copy()
    if "rhythmic" not in fits and "rhythmic_lombscargle" in fits:
        fits["rhythmic"] = fits["rhythmic_lombscargle"]
    if "best_period_hours" not in fits and "lombscargle_period_hours" in fits:
        fits["best_period_hours"] = fits["lombscargle_period_hours"]
    needed = {"identity", "rhythmic", "best_period_hours"}
    missing = needed - set(fits)
    if missing:
        raise SystemExit(
            "--line-dash-metric rhythm_period_band needs rhythms.csv columns: "
            + ", ".join(sorted(missing))
        )
    fits = fits.drop_duplicates("identity", keep="last").set_index("identity")
    fitted = fits.reindex(identities)
    raw_status = fitted["rhythmic"]
    if raw_status.dtype.kind in "bif":
        passed = raw_status.fillna(False).astype(bool)
    else:
        passed = raw_status.astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes", "rhythmic"}
        )
    period = pd.to_numeric(fitted["best_period_hours"], errors="coerce")

    labels = [f"{low:g}-{high:g} h rhythmic"
              for low, high in zip(edges[:-1], edges[1:])]
    under = f"< {edges[0]:g} h rhythmic"
    over = f"> {edges[-1]:g} h rhythmic"
    unavailable = "Rhythmic; period unavailable"
    unknown = "Rhythm status unknown"
    not_rhythmic = "Not rhythmic"
    no_result = "No rhythm result"
    categories = pd.Series(no_result, index=identities, dtype=object)
    has_result = pd.Series(identities.isin(fits.index), index=identities)
    categories.loc[has_result & raw_status.isna()] = unknown
    categories.loc[has_result & raw_status.notna() & ~passed] = not_rhythmic
    categories.loc[passed & ~np.isfinite(period)] = unavailable
    categories.loc[passed & (period < edges[0])] = under
    categories.loc[passed & (period > edges[-1])] = over
    for index, (low, high, label) in enumerate(zip(edges[:-1], edges[1:], labels)):
        above_low = period.ge(low) if index == 0 else period.gt(low)
        categories.loc[passed & above_low & period.le(high)] = label

    desired_order = [under, *labels, over, unavailable, not_rhythmic, unknown, no_result]
    order = [label for label in desired_order if label in set(categories)]
    method = "lomb"
    if "primary_rhythm_test" in fits and fits["primary_rhythm_test"].notna().any():
        method = str(fits["primary_rhythm_test"].dropna().mode().iloc[0])
    method_label = workbench.PERIOD_METHODS.get(
        method, {"label": RHYTHM_METHOD_LABELS.get(method, method)})["label"]
    estimator = method
    if "period_estimation_method" in fits and fits["period_estimation_method"].notna().any():
        estimator = str(fits["period_estimation_method"].dropna().mode().iloc[0])
    estimator_label = (
        str(fits["best_method_label"].dropna().mode().iloc[0])
        if "best_method_label" in fits and fits["best_method_label"].notna().any()
        else workbench.PERIOD_METHODS.get(
            estimator, {"label": RHYTHM_METHOD_LABELS.get(estimator, estimator)})["label"]
    )
    details = pd.DataFrame({
        "identity": identities,
        "rhythmic": fitted["rhythmic"].to_numpy(),
        "best_period_hours": period.to_numpy(),
        "line_dash_category": categories.to_numpy(),
        "rhythm_metric": rhythm_metric,
        "rhythm_test": method_label,
        "period_estimator": estimator_label,
    })
    return categories, order, details, method_label, estimator_label


def _summary_dash_categories(
    values: pd.Series,
) -> tuple[pd.Series, list[str], pd.DataFrame]:
    """Turn a summary column into a finite dash vocabulary without hiding missing cells."""
    categories = pd.Series("Missing", index=values.index, dtype=object)
    numeric = pd.to_numeric(values, errors="coerce")
    originally_numeric = values.dtype.kind in "bif" or numeric.notna().sum() == values.notna().sum()
    if originally_numeric and numeric.notna().any():
        finite = numeric[np.isfinite(numeric)]
        if finite.nunique() == 1:
            categories.loc[finite.index] = f"One value ({finite.iloc[0]:g})"
        elif values.dtype.kind == "b":
            categories.loc[finite.index] = np.where(finite.astype(bool), "True", "False")
        else:
            low, high = map(float, finite.quantile([1 / 3, 2 / 3]))
            categories.loc[finite.index[finite <= low]] = "Low third"
            categories.loc[finite.index[(finite > low) & (finite <= high)]] = "Middle third"
            categories.loc[finite.index[finite > high]] = "High third"
    else:
        categories.loc[values.notna()] = values.loc[values.notna()].astype(str)

    order = list(dict.fromkeys(categories.tolist()))
    details = pd.DataFrame({
        "identity": values.index,
        "line_dash_source_value": values.to_numpy(),
        "line_dash_category": categories.to_numpy(),
    })
    return categories, order, details


def _dash_styles(order: list[str]) -> dict[str, object]:
    if len(order) > len(DASH_PATTERNS):
        raise SystemExit(
            f"--line-dash-metric produced {len(order)} categories; at most "
            f"{len(DASH_PATTERNS)} can be distinguished. Use a grouped column or fewer period bins."
        )
    return dict(zip(order, DASH_PATTERNS))


def _format_value(value: float, unit: str) -> str:
    number = f"{value:.3g}"
    return f"{number} {unit}" if unit else number


def _summary_label(column: str) -> str:
    """Name the per-cell summary operation that ``describe`` deliberately strips."""
    label = describe(column).label
    for suffix, statistic in (("_median", "Median"), ("_mean", "Mean")):
        if column.endswith(suffix):
            return f"{statistic} {label.lower()}"
    return label


@figure(
    number=8,
    slug="identity-trajectories",
    summary="where every cell went over the whole recording",
    title="Centroid path of every cell over {span} h",
    grammar="trajectory map with independent colour width and dash encodings",
    reads=(Table("cell_frame.csv", module="motility"),
           Table("cell_summary.csv", module="motility"),
           Table("rhythms.csv", module="rhythms", optional=True)),
    panels=(Panel("map", motility_panels.trajectory_map,
                  title="Every centroid path"),),
    options=(
        Option("metrics", default="net_displacement_px", cast=str, metavar="COL",
               help="which measured column colours each path"),
        Option("trace_luts", default=None, cast=str, metavar="NAME",
               help="the colour map each path's colour runs along"),
        Option("line_width_metric", default="", cast=str),
        Option("line_width_range", default=[0.55, 2.8], cast=numbers),
        Option("line_dash_metric", default="", cast=str),
        Option("rhythm_metric", default="corrected_mean"),
        Option("period_bins", default=DEFAULT_PERIOD_BINS, cast=numbers),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    colour_by = ctx.option("metrics")
    path_lut = ctx.option("trace_luts")
    width_by = str(ctx.option("line_width_metric")).strip()
    width_range = list(ctx.option("line_width_range"))
    dash_by = str(ctx.option("line_dash_metric")).strip()
    rhythm_metric = str(ctx.option("rhythm_metric"))
    period_bins = list(ctx.option("period_bins"))

    ctx.panels()
    cell_frame = ctx.table("cell_frame.csv")
    tracks = ctx.table("cell_summary.csv").set_index("identity")

    colour_values = _metric_values(tracks, colour_by, "metrics")

    path = cell_frame[
        ["identity", "frame_index", "hours", "centroid_x", "centroid_y"]
    ].dropna(subset=["centroid_x", "centroid_y"]).sort_values(
        ["identity", "frame_index"]).copy()
    path[colour_by] = path["identity"].map(colour_values)
    if "total_path_px" in tracks.columns:
        path["total_path_px"] = path["identity"].map(tracks["total_path_px"])

    values = colour_values.dropna()
    cells = int(path["identity"].nunique())
    label = describe(colour_by)

    identities = pd.Index(path["identity"].drop_duplicates().tolist(), name="identity")
    base_width = ctx.theme.stroke("line") * 0.55
    widths = pd.Series(base_width, index=tracks.index, dtype=float)
    width_legend: list[tuple[str, float, float]] = []
    width_label = None
    if width_by:
        width_values = _metric_values(tracks, width_by, "line_width_metric")
        widths, width_legend = _scaled_line_widths(
            width_values, base_width, width_range
        )
        width_label = describe(width_by)
        path[width_by] = path["identity"].map(width_values)
    path["line_width"] = path["identity"].map(widths)

    dash_categories = pd.Series("All trajectories", index=tracks.index, dtype=object)
    dash_order = ["All trajectories"]
    dash_details = pd.DataFrame({"identity": tracks.index})
    dash_title = None
    rhythm_method = None
    rhythm_estimator = None
    if dash_by == "rhythm_period_band":
        rhythms = ctx.optional_table("rhythms.csv")
        if rhythms is None:
            raise SystemExit(
                "--line-dash-metric rhythm_period_band needs rhythms.csv in this run"
            )
        dash_categories, dash_order, dash_details, rhythm_method, rhythm_estimator = (
            _rhythm_period_categories(
                rhythms, tracks.index, rhythm_metric, period_bins
            )
        )
        dash_title = f"Dash: {describe(rhythm_metric).label.lower()} rhythm result"
    elif dash_by:
        if dash_by not in tracks:
            available = ", ".join(map(str, tracks.columns))
            raise SystemExit(
                f"--line-dash-metric {dash_by} is not in cell_summary.csv. "
                f"Available columns: {available}"
            )
        dash_categories, dash_order, dash_details = _summary_dash_categories(
            tracks[dash_by]
        )
        dash_title = f"Dash: {describe(dash_by).label}"
    dash_styles = _dash_styles(dash_order)
    path["line_dash_category"] = path["identity"].map(dash_categories)

    encoded = bool(width_by or dash_by)
    figure_ = ctx.sheet(17.6 if encoded else 14.6, 12.8)
    ax = figure_.add_axes(
        [0.079, 0.190, 0.581, 0.640] if encoded
        else [0.095, 0.190, 0.700, 0.640]
    )

    grouped = list(path.groupby("identity", sort=True))
    groups = [
        (group["frame_index"].to_numpy(int),
         group[["centroid_x", "centroid_y"]].to_numpy(float),
         float(tracks[colour_by].get(identity, float("nan"))))
        for identity, group in grouped
    ]
    group_identities = [identity for identity, _ in grouped]
    drawn = ctx.drew("map", motility_panels.trajectory_map(
        ax, groups, ctx.theme,
        field=ctx.field,
        look=resolve_look(ctx.theme, path_lut) if path_lut else None,
        vmax=float(values.max()),
        identities=group_identities,
        line_widths=[float(widths.get(identity, base_width))
                     for identity in group_identities],
        line_dash_categories=[str(dash_categories.get(identity, "Missing"))
                              for identity in group_identities],
        line_dash_styles=dash_styles,
    ))
    collection = drawn.extra["collection"]
    breaks = drawn.extra["breaks"]
    colour_bar(
        figure_, collection,
        [0.692, 0.270, 0.014, 0.440] if encoded
        else [0.832, 0.255, 0.018, 0.470],
        ctx.theme,
        label=(f"{label.label} over the movie"
               + (f" ({label.unit})" if label.unit else "")),
    )

    if width_by and width_label is not None:
        width_handles = [
            Line2D([], [], color=ctx.theme.colour("ink"), linewidth=width)
            for _, _, width in width_legend
        ]
        width_unit = "pixels" if width_by.startswith("area_px") else width_label.unit
        width_labels = [
            f"{name}: {_format_value(value, width_unit)}"
            for name, value, _ in width_legend
        ]
        figure_.legend(
            handles=width_handles, labels=width_labels,
            title=f"Line width: {_summary_label(width_by)}",
            loc="upper left", bbox_to_anchor=(0.742, 0.815),
            frameon=False, fontsize=ctx.theme.size("caption"),
            title_fontsize=ctx.theme.size("subtitle"), handlelength=3.2,
        )
    if dash_by and dash_title is not None:
        dash_handles = [
            Line2D([], [], color=ctx.theme.colour("ink"),
                   linewidth=ctx.theme.stroke("emphasis"),
                   linestyle=dash_styles[category])
            for category in dash_order
        ]
        figure_.legend(
            handles=dash_handles, labels=dash_order, title=dash_title,
            loc="upper left", bbox_to_anchor=(0.742, 0.555),
            frameon=False, fontsize=ctx.theme.size("caption"),
            title_fontsize=ctx.theme.size("subtitle"), handlelength=3.2,
        )

    figure_data = drawn.data.rename(columns={"value": "colour_value"}).copy()
    figure_data["colour_metric"] = colour_by
    figure_data["line_width_metric"] = width_by or "fixed"
    figure_data["line_dash_metric"] = dash_by or "fixed"
    if width_by:
        figure_data["line_width_source_value"] = figure_data["identity"].map(
            tracks[width_by]
        )
    if not dash_details.empty:
        figure_data = figure_data.merge(
            dash_details.drop(columns=["line_dash_category"], errors="ignore")
            .drop_duplicates("identity"),
            on="identity", how="left",
        )

    width_sentence = (
        f" Line width is {_summary_label(width_by).lower()}." if width_label else ""
    )
    dash_sentence = (
        f" Dash is the {describe(rhythm_metric).label.lower()} rhythm-period band."
        if dash_by == "rhythm_period_band"
        else f" Dash is {describe(dash_by).label.lower()}." if dash_by else ""
    )
    rhythm_note = (
        f" Dash rhythm status comes from {rhythm_method}; periods come from "
        f"{rhythm_estimator}, with band boundaries at "
        f"{', '.join(f'{value:g}' for value in period_bins)} h."
        if rhythm_method else ""
    )

    return FigureResult(
        figure=figure_,
        axes=[ax],
        # The map is bounded by the field, so it keeps all four edges: a field
        # with two sides missing reads as a plot that happens to stop there.
        keep_spines=("top", "right"),
        figure_data=figure_data,
        auxiliary={"trajectory_points.csv": path},
        heading="Identity trajectories",
        title_fields={"span": f"{ctx.summary['hours_covered']:.0f}"},
        subtitle=(
            f"{ctx.summary['stem']}, {cells} identities across a "
            f"{ctx.field['width']} x {ctx.field['height']} px field. One path per "
            f"cell, from its centroid in\nevery frame it holds; the dot marks first "
            f"appearance. Colour is {label.label.lower()}: median "
            f"{values.median():.1f} {label.unit or ''}, largest "
            f"{values.max():.0f} {label.unit or ''}.{width_sentence}{dash_sentence}"
        ),
        footnote=ctx.footnote(
            f"The path is cut at each of the {breaks} gaps rather than bridged across "
            f"one, so a long straight stroke is a movement the movie showed."
            + rhythm_note
        ),
        readme=f"""## What the figure shows

Every identity's centroid path across the whole recording, drawn in the movie's
own pixel coordinates with y counting downwards, so the map is oriented the way
the frames are. Stationary cells appear as tight knots; the mobile minority draw
long strokes. Colour is `{colour_by}`. Line width is `{width_by or 'fixed'}` and
dash is `{dash_by or 'fixed'}`; these are independent, so changing one never
changes either of the others.

For numeric summary columns, dash uses labelled low, middle and high thirds.
For categorical columns it uses the values themselves. The special
`rhythm_period_band` value reads `{rhythm_metric}` from `rhythms.csv`, assigns
only cells that passed the primary rhythm test to the hour bands bounded by
`{','.join(f'{value:g}' for value in period_bins)}`, and keeps separate
categories for a failed test and no rhythm result. This can therefore separate
a significant 6-hour cell from a significant 24-hour cell without assuming
that either period exists in the current run.

The axes are the field, not the data. A map bounded by wherever the cells
happen to be changes magnification between runs and invites two maps to be
compared at different scales without saying so.

## What it is a read-out of

Tracking, not a separate analysis: the centroids come from `cell_frame.csv` and
the colour and optional line width or generic dash measurement from
`cell_summary.csv`. Rhythm-period dashes use `rhythms.csv` only when requested.
Paths are cut at gaps rather than bridged
across them, because a bridged gap draws a straight stroke that reads as
evidence of travel the movie never showed. The cutting is done by
`panels.motility.path_segments`, which the step histogram's own rule matches.""",
        console=(f"cells: {cells}  breaks: {breaks}  colour: {colour_by}  "
                 f"width: {width_by or 'fixed'}  dash: {dash_by or 'fixed'}"),
    )


if __name__ == "__main__":
    run_figure("identity-trajectories", DEFAULT_RUN)
