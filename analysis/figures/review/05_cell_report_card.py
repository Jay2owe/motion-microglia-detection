"""Figure 5: the per-cell report card the package can emit for every identity.

Every part of the page is an argument. The panels themselves live in ``panels``
and are drawn by this file, not defined in it, so the same image strip and the
same trace can be used on a page of their own::

    python analysis/figures/05_cell_report_card.py <run> --identity 7
    ... --metrics corrected_mean,area_px,solidity,step_px_gapless
    ... --images 6                       six tiles, evenly spaced
    ... --image-hours 0,12,24,36,48      or exactly these hours
    ... --images 0                       no tiles at all
    ... --cell-lut dluc_purple           the LUT the cell is displayed through
    ... --image-filter auto-organotypic  display-only grain suppression
    ... --display-gain 6                 magnify temporal differences six-fold
    ... --trace-layout overlay           compare trace shapes on one axis
    ... --trace-view detrended           show the selected detrending
    ... --detrend running_mean           a Circadian Workbench detrending method
    ... --fit-method lomb                period estimator used for fitted curves
    ... --trace-luts reporter,viridis,#c0392b
    ... --outline white
    ... --fit corrected_mean,area_px     or `--fit all`, or `--fit=` for none

The layout follows what was asked for: the canvas grows a row per metric and
loses the strip entirely when no tiles are wanted.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parent.parent))

import numpy as np
import pandas as pd
import tifffile

from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from _metrics import axis_label, describe, role_for
from _options import commas
from _schema import (FigureContext, FigureResult, Input, Option, Panel, Table,
                     figure, run_figure)
from panels import (Look, harmonic_curve, image_strip, resolve_look, trace,
                    trace_overlay)
from panels import intensity

DEFAULT_RUN = "outputs/a01_95_A3_accepted_baseline"


@figure(
    number=5,
    slug="cell-report-card",
    purpose="review",
    summary="per-cell image tiles and measurement traces for identity review",
    title="Cell {identity}: {names}",
    grammar="image strip over one trace per requested measurement, for one cell",
    reads=(Table("cell_frame.csv", module="motility"),
           Table("rhythms.csv", module="rhythms"),
           Input("labels"), Input("raw")),
    panels=(
        Panel("tiles", image_strip, block=True, needs=("labels", "raw"),
              item_width_inches=2.5, item_height_inches=2.5,
              title="The outline over the signal"),
        Panel("traces", trace, title="One trace per measurement"),
    ),
    options=(
        # This page is a layout for any cell, not a chosen result, so which cell
        # it draws is an argument. 44 is only the one that happens to be
        # complete.
        Option("identity", default=44),
        Option("metrics",
               default=["corrected_mean", "area_px", "turnover_index"]),
        Option("images", default=10),
        Option("image_hours", default=[]),
        Option("cell_lut", default="dluc_purple"),
        Option("image_filter", default="auto-organotypic"),
        Option("display_black_percentile", default=50.0),
        Option("display_white_percentile", default=99.8),
        Option("display_range_scope", default="stack"),
        Option("display_gamma", default=0.7),
        Option("display_gain", default=6.0),
        Option("display_spatial_sigma", default=1.0),
        Option("display_pool_px", default=4.0),
        Option("display_sharpness", default=3.0),
        Option("display_noise_multiple", default=1.0),
        Option("display_pad_frames", default=64),
        Option("trace_luts", default=[]),
        Option("trace_layout", default="stack"),
        Option("trace_view", default="raw"),
        Option("outline", default="outline"),
        Option("fit", default="", cast=str, metavar="COL,COL|all",
               help="which traces carry an explicit descriptive Workbench cosinor: "
                    "a metric list, `all`, or empty for none"),
        Option("fit_method", default=None),
        Option("significance_method", default=None),
        Option("period_config", default={}),
        Option("period_min_hours", default=2.0),
        Option("period_max_hours", default=48.0),
        Option("detrend", default=None),
        Option("detrend_window_hours", default=None),
        Option("hour_ticks", default=None),
    ),
)
def build(ctx: FigureContext) -> FigureResult:
    identity = ctx.option("identity")
    hour_ticks = ctx.option("hour_ticks")
    metrics = ctx.option("metrics")
    n_tiles = ctx.option("images")
    tile_hours = ctx.option("image_hours")
    cell_lut = ctx.option("cell_lut")
    image_filter = ctx.option("image_filter")
    display_black = ctx.option("display_black_percentile")
    display_white = ctx.option("display_white_percentile")
    display_range_scope = str(ctx.option("display_range_scope")).lower()
    display_gamma = ctx.option("display_gamma")
    display_gain = ctx.option("display_gain")
    display_spatial_sigma = ctx.option("display_spatial_sigma")
    display_pool_px = ctx.option("display_pool_px")
    display_sharpness = ctx.option("display_sharpness")
    display_noise_multiple = ctx.option("display_noise_multiple")
    display_pad_frames = ctx.option("display_pad_frames")
    trace_luts = ctx.option("trace_luts")
    trace_layout = str(ctx.option("trace_layout")).lower()
    trace_view = str(ctx.option("trace_view")).lower()
    outline = ctx.option("outline")
    fit = ctx.option("fit")
    inherited = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    resolved = workbench.resolve_analysis_options(inherited, ctx.option)
    rhythm_params = resolved["params"]
    detrending = {
        name: resolved[name] for name in workbench.DETREND_DEFAULTS
    }
    detrend = str(detrending["detrend"])
    fit_method = resolved["method"]
    significance_method = resolved["significance_method"]

    if trace_layout not in ("stack", "overlay"):
        raise SystemExit("--trace-layout must be stack or overlay")
    if trace_view not in ("raw", "detrended"):
        raise SystemExit("--trace-view must be raw or detrended")
    if detrend not in workbench.DETREND_METHODS:
        raise SystemExit(
            f"--detrend {detrend!r} is unknown; choose "
            + ", ".join(workbench.DETREND_METHODS))
    if display_range_scope not in ("stack", "displayed"):
        raise SystemExit("--display-range-scope must be stack or displayed")
    if fit_method not in workbench.PERIOD_METHODS:
        raise SystemExit(
            f"--fit-method {fit_method!r} is unknown; choose "
            + ", ".join(workbench.PERIOD_METHODS))

    if not metrics:
        raise SystemExit(
            "--metrics needs at least one column; pass --images 0 to drop the tiles")

    panels = ctx.panels()
    cell_frame = ctx.table("cell_frame.csv")
    rhythms = ctx.table("rhythms.csv")

    unknown = [column for column in metrics if column not in cell_frame.columns]
    if unknown:
        traceable = ", ".join(
            c for c in cell_frame.columns
            if c not in ("stem", "condition", "subject", "identity")
            and cell_frame[c].dtype.kind in "fi"
        )
        raise SystemExit(
            f"--metrics names {', '.join(unknown)}, which cell_frame.csv does not "
            f"have.\nColumns available: {traceable}"
        )

    cell = cell_frame[
        cell_frame["identity"] == identity].sort_values("frame_index").copy()
    if cell.empty:
        present = sorted(cell_frame["identity"].unique())
        raise SystemExit(
            f"identity {identity} is not in this run; it holds {len(present)} "
            f"identities, {present[0]} to {present[-1]}"
        )
    stored_fits = rhythms[rhythms["identity"] == identity].copy()

    # Which traces carry the model curve. The period is re-estimated through
    # Circadian Workbench at the selected detrending, so this control is not
    # tied to whichever method produced the run's summary table.
    if not str(fit).strip():
        fitted_metrics = []
    elif fit.strip().lower() == "all":
        fitted_metrics = list(metrics)
    else:
        fitted_metrics = commas(fit)
        missing_fit = [column for column in fitted_metrics if column not in metrics]
        if missing_fit:
            raise SystemExit(
                "--fit can only name traces selected by --metrics; missing: "
                + ", ".join(missing_fit))

    hours = cell["hours"].to_numpy(float)

    detrended_by_metric: dict[str, np.ndarray] = {}
    baseline_by_metric: dict[str, np.ndarray] = {}
    fit_evidence: dict[str, dict] = {}
    for column in metrics:
        raw_values = cell[column].to_numpy(float)
        usable = np.isfinite(hours) & np.isfinite(raw_values)
        detrended_values = np.full(raw_values.shape, np.nan, dtype=float)
        baseline_values = np.full(raw_values.shape, np.nan, dtype=float)
        if usable.sum() >= 2:
            result = workbench.detrend_trace(
                hours[usable], raw_values[usable], rhythm_params, method=detrend)
            detrended_values[usable] = np.asarray(result["values"], dtype=float)
            baseline_values[usable] = np.asarray(result["baseline"], dtype=float)
        detrended_by_metric[column] = detrended_values
        baseline_by_metric[column] = baseline_values
        if column in fitted_metrics and usable.sum() >= resolved["min_observations"]:
            try:
                fit_evidence[column] = workbench.estimate_one(
                    hours[usable], raw_values[usable], rhythm_params,
                    fit_method, detrend=detrend,
                )
                result = fit_evidence[column]
                evidence = result if significance_method == fit_method else workbench.estimate_one(
                    hours[usable], raw_values[usable], rhythm_params,
                    significance_method, detrend=detrend)
                result.update(
                    significance_method=significance_method,
                    significance_p_value=evidence.get("p_value"),
                    significance_period_hours=evidence.get("period_hours"),
                    significance_status=evidence.get("status", "failed"))
                period = result.get("period_hours")
                span = float(hours[usable].max() - hours[usable].min())
                cycles = (
                    span / float(period)
                    if period is not None and np.isfinite(period) and float(period) > 0
                    else np.nan
                )
                result.update(
                    cycles_observed=cycles,
                    period_underdetermined=(
                        not np.isfinite(cycles) or cycles < resolved["min_cycles"]
                    ),
                )
            except ValueError as error:
                fit_evidence[column] = {
                    "method": fit_method,
                    "method_label": workbench.PERIOD_METHODS[fit_method]["label"],
                    "status": "failed",
                    "rhythm_status": "unknown",
                    "period_hours": np.nan,
                    "p_value": np.nan,
                    "message": str(error),
                    "workbench_version": workbench.WORKBENCH_VERSION,
                }
        elif column in fitted_metrics:
            fit_evidence[column] = {
                "method": fit_method,
                "method_label": workbench.PERIOD_METHODS[fit_method]["label"],
                "status": "not_tested",
                "rhythm_status": "not tested",
                "period_hours": np.nan,
                "p_value": np.nan,
                "message": "too_few_observations",
                "workbench_version": workbench.WORKBENCH_VERSION,
            }

    fitted_columns = list(fit_evidence)
    adjusted = workbench.adjust_pvalues(
        [fit_evidence[column].get("significance_p_value", np.nan)
         for column in fitted_columns],
        resolved["multiple_testing"],
    )
    for column, q_value in zip(fitted_columns, adjusted):
        result = fit_evidence[column]
        result.update(
            q_value=q_value,
            correction=resolved["multiple_testing"],
            alpha=resolved["rhythmic_alpha"],
        )
        tested = result.get("significance_status") == "ok" and np.isfinite(q_value)
        result["rhythm_status"] = (
            "rhythmic" if tested and q_value < resolved["rhythmic_alpha"]
            else "not rhythmic" if tested else "not tested"
        )

    # Label frame k is source frame k + offset; the tables carry both, so the offset
    # is read off them rather than repeated here.
    offset = int(cell["source_imagej_frame"].iloc[0] - cell["imagej_frame"].iloc[0])
    frames = cell["frame_index"].to_numpy(int)

    tile_frames: list[int] = []
    if "tiles" in panels and (n_tiles > 0 or tile_hours):
        if tile_hours:
            # Nearest observed frame to each requested hour: an hour the cell was
            # missing for still gets a tile, and the tile says which hour it is.
            tile_frames = [int(frames[int(np.abs(hours - wanted).argmin())])
                           for wanted in tile_hours]
        else:
            picks = np.linspace(0, len(frames) - 1, min(n_tiles, len(frames)))
            tile_frames = [int(frames[i]) for i in picks.round().astype(int)]

    crops: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    display_settings: dict = {}
    crop_box = (0, 0, 0, 0)
    if tile_frames:
        labels = tifffile.imread(ctx.input_path("labels"))
        raw_full = tifffile.imread(ctx.input_path("raw"))
        raw = raw_full[offset: offset + labels.shape[0]]

        # One crop for the whole strip, big enough for the cell at its largest, so
        # a tile that looks bigger is a bigger cell and not a tighter crop.
        centre_y = int(round(cell["centroid_y"].median()))
        centre_x = int(round(cell["centroid_x"].median()))
        half = 0
        for frame_index in frames:
            ys, xs = np.nonzero(labels[frame_index] == identity)
            if ys.size:
                half = max(half, int(np.abs(ys - centre_y).max()),
                           int(np.abs(xs - centre_x).max()))
        half = min(half + 5, 40)
        top = max(centre_y - half, 0)
        left_edge = max(centre_x - half, 0)
        bottom = min(centre_y + half + 1, labels.shape[1])
        right = min(centre_x + half + 1, labels.shape[2])
        crop_box = (top, bottom, left_edge, right)

        display_stack, display_settings = intensity.presentation_stack(
            raw[:, top:bottom, left_edge:right],
            image_filter=image_filter,
            black_percentile=display_black,
            white_percentile=display_white,
            gamma=display_gamma,
            time_gain=display_gain,
            spatial_sigma_px=display_spatial_sigma,
            pool_px=display_pool_px,
            sharpness=display_sharpness,
            noise_multiple=display_noise_multiple,
            pad_frames=display_pad_frames,
            range_frames=tile_frames if display_range_scope == "displayed" else None,
        )
        crops = [display_stack[f] for f in tile_frames]
        masks = [labels[f][top:bottom, left_edge:right] == identity for f in tile_frames]

    header_in = 1.3                       # title and optional explicit subtitle
    panel_in = 2.25                       # one trace
    overlay_in = 4.25                     # overlaid traces need one full panel
    gap_in = 0.60                         # between traces
    foot_in = 1.5                         # tick labels, x-axis label, optional footnote
    tile_title_in = 0.55                  # the hour written over each tile
    width_in = 15.5
    left, plot_width = 0.105, 0.845
    tile_panel = ctx.spec.panel("tiles")
    if tile_frames:
        required_width, _ = tile_panel.grid_minimum(
            len(tile_frames), len(tile_frames))
        width_in = max(width_in, required_width / plot_width)

    # A tile keeps its declared width when more frames are requested; the page
    # widens and the square tile's height follows that preserved width.
    tile_in = ((plot_width * width_in
                - tile_panel.item_gap_inches * (len(tile_frames) - 1))
               / len(tile_frames)) if tile_frames else 0.0
    strip_in = (tile_in + tile_title_in + 0.55) if tile_frames else 0.0
    traces_in = (overlay_in if trace_layout == "overlay" else
                 len(metrics) * panel_in + (len(metrics) - 1) * gap_in)
    height_in = header_in + strip_in + traces_in + foot_in
    figure_ = ctx.sheet(width_in, height_in)

    def fraction(inches: float) -> float:
        return inches / height_in

    tile_axes, vmin, vmax = [], float("nan"), float("nan")
    if tile_frames:
        strip = ctx.drew("tiles", image_strip(
            figure_,
            (left, fraction(height_in - header_in - tile_title_in - tile_in),
             plot_width, fraction(tile_in)),
            crops, ctx.theme,
            masks=masks,
            titles=[f"{float(cell.loc[cell.frame_index == f, 'hours'].iloc[0]):.0f} h"
                    for f in tile_frames],
            look=resolve_look(ctx.theme, cell_lut, "morphology") if cell_lut else None,
            outline_role=outline,
            gap=tile_panel.item_gap_inches / width_in,
            vmin=0.0, vmax=1.0,
        ))
        tile_axes = strip.axes
        vmin, vmax = strip.extra["vmin"], strip.extra["vmax"]

    looks = {
        column: (
            resolve_look(ctx.theme, trace_luts[index], role_for(column))
            if index < len(trace_luts)
            else Look(colour=ctx.theme.colour(role_for(column)))
        )
        for index, column in enumerate(metrics)
    }

    shown_by_metric = {
        column: (cell[column].to_numpy(float) if trace_view == "raw"
                 else detrended_by_metric[column])
        for column in metrics
    }
    plotted_by_metric: dict[str, np.ndarray] = {}
    fitted_by_metric = {
        column: np.full(hours.shape, np.nan, dtype=float) for column in metrics
    }
    axes = []
    if trace_layout == "overlay":
        ax = figure_.add_axes([
            left, fraction(foot_in), plot_width, fraction(overlay_in)])
        overlaid = ctx.drew("traces", trace_overlay(
            ax, hours, shown_by_metric, ctx.theme, looks=looks,
            labels={column: describe(column).label for column in metrics},
            normalise="z", hour_ticks=hour_ticks,
        ))
        for column in metrics:
            plotted_by_metric[column] = overlaid.data.loc[
                overlaid.data["series"] == column, "value"].to_numpy(float)
        axes.append(ax)
    else:
        each = fraction(panel_in)
        for index, column in enumerate(metrics):
            top_down = len(metrics) - 1 - index
            ax = figure_.add_axes([
                left,
                fraction(foot_in + top_down * (panel_in + gap_in)),
                plot_width,
                each,
            ])
            plotted_by_metric[column] = shown_by_metric[column]
            y_label = axis_label(column, ctx.interval)
            if trace_view == "detrended":
                y_label = "Detrended " + y_label[0].lower() + y_label[1:]
            trace(
                ax, hours, plotted_by_metric[column], ctx.theme,
                look=looks[column], hour_ticks=hour_ticks,
                show_x=(index == len(metrics) - 1), y_label=y_label,
            )
            axes.append(ax)

    fit_notes = []
    fitted_curves = 0
    for column in fitted_metrics:
        result = fit_evidence.get(column, {})
        period = result.get("period_hours")
        period = float(period) if period is not None else np.nan
        if not np.isfinite(period):
            fit_notes.append(f"{describe(column).label}: fit unavailable")
            continue
        target_ax = axes[0] if trace_layout == "overlay" else axes[metrics.index(column)]
        fit_values = plotted_by_metric[column]
        fit_baseline = None
        if trace_view == "raw":
            fit_values = detrended_by_metric[column]
            fit_baseline = baseline_by_metric[column]
            if trace_layout == "overlay":
                raw_values = cell[column].to_numpy(float)
                centre = float(np.nanmean(raw_values))
                spread = float(np.nanstd(raw_values))
                if np.isfinite(spread) and spread > 0:
                    fit_values = fit_values / spread
                    fit_baseline = (fit_baseline - centre) / spread
                else:
                    fit_values = np.zeros_like(fit_values)
                    fit_baseline = np.zeros_like(fit_baseline)
        if fit_method == "fft_nlls":
            fitted = workbench.fft_nlls_fitted_values(hours, result)
            if trace_view == "raw":
                fitted = fitted + baseline_by_metric[column]
            if trace_layout == "overlay":
                reference = shown_by_metric[column]
                centre, spread = float(np.nanmean(reference)), float(np.nanstd(reference))
                fitted = (fitted - centre) / spread if spread > 0 else np.zeros_like(fitted)
            fitted_by_metric[column] = fitted
            target_ax.plot(
                hours, fitted, color=looks[column].colour or ctx.theme.colour(role_for(column)),
                linewidth=ctx.theme.stroke("emphasis"), linestyle=(0, (6, 3)))
        else:
            fitted_by_metric[column] = harmonic_curve(
                target_ax, hours, fit_values, ctx.theme,
                period_hours=period, baseline=fit_baseline,
                colour=(looks[column].colour or ctx.theme.colour(role_for(column))),
            ).extra["fitted"]
        fitted_curves += 1
        q_value = result.get("q_value")
        p_text = (f", {significance_method} q = {float(q_value):.3g}"
                  if q_value is not None and np.isfinite(q_value)
                  else ", no corrected significance value")
        status = str(result.get("rhythm_status") or "unknown")
        note = (f"{describe(column).label}: {period:.2f} h, "
                f"{result.get('method_label', fit_method)}{p_text}, {status}")
        fit_notes.append(note)
        if trace_layout == "stack":
            target_ax.text(
                1.0, 1.04, note, transform=target_ax.transAxes,
                ha="right", va="bottom", fontsize=ctx.theme.size("caption"),
                color=ctx.theme.colour("caption"),
            )
    if trace_layout == "overlay" and fit_notes:
        axes[0].text(
            1.0, 1.04,
            ("Dashed fits · " if fitted_curves else "Period tests · ")
            + " | ".join(fit_notes),
            transform=axes[0].transAxes, ha="right", va="bottom",
            fontsize=ctx.theme.size("caption"), color=ctx.theme.colour("caption"),
        )

    rows = []
    identifiers = ["frame_index", "imagej_frame", "source_imagej_frame"]
    for position, source_row in cell.reset_index(drop=True).iterrows():
        for column in metrics:
            evidence = fit_evidence.get(column, {})
            rows.append({
                "identity": int(identity),
                **{name: source_row[name] for name in identifiers},
                "hours": float(hours[position]),
                "metric": column,
                "raw_value": float(source_row[column]),
                "detrended_value": float(detrended_by_metric[column][position]),
                "baseline_value": float(baseline_by_metric[column][position]),
                "plotted_value": float(plotted_by_metric[column][position]),
                "fitted_value": float(fitted_by_metric[column][position]),
                "trace_layout": trace_layout,
                "trace_view": trace_view,
                "detrend": detrend,
                "detrend_window_hours": detrending["detrend_window_hours"],
                "fit_method": fit_method if column in fitted_metrics else "",
                "fit_period_hours": evidence.get("period_hours", np.nan),
                "fit_p_value": evidence.get("p_value", np.nan),
                "significance_method": evidence.get("significance_method", ""),
                "significance_p_value": evidence.get("significance_p_value", np.nan),
                "significance_q_value": evidence.get("q_value", np.nan),
                "significance_correction": evidence.get("correction", ""),
                "significance_period_hours": evidence.get("significance_period_hours", np.nan),
                "fit_relative_amplitude_error": evidence.get("rae", np.nan),
                "rhythm_status": evidence.get("rhythm_status", ""),
            })
    figure_data = pd.DataFrame(rows)

    evidence_rows = []
    for column, result in fit_evidence.items():
        evidence_rows.append({
            "metric": column,
            "method": result.get("method", fit_method),
            "method_label": result.get("method_label", fit_method),
            "detrend": detrend,
            "detrend_window_hours": detrending["detrend_window_hours"],
            "status": result.get("status", "unknown"),
            "rhythm_status": result.get("rhythm_status", "unknown"),
            "period_hours": result.get("period_hours", np.nan),
            "period_error_hours": result.get("period_error_hours", np.nan),
            "significance_method": result.get("significance_method"),
            "significance_p_value": result.get("significance_p_value"),
            "q_value": result.get("q_value"),
            "correction": result.get("correction"),
            "significance_period_hours": result.get("significance_period_hours"),
            "cycles_observed": result.get("cycles_observed"),
            "period_underdetermined": result.get("period_underdetermined"),
            **{key: result.get(key) for key in ("phase_hours", "phase_error_hours",
                "amplitude", "amplitude_error", "rae", "components", "diagnostics")},
            "p_value": result.get("p_value", np.nan),
            "alpha": result.get("alpha", np.nan),
            "goodness_of_fit": result.get("goodness_of_fit", np.nan),
            "workbench_version": result.get(
                "workbench_version", workbench.WORKBENCH_VERSION),
            "message": result.get("message", ""),
        })
    auxiliary = {
        "stored_rhythm_fits_this_cell.csv": stored_fits,
        "selected_fit_evidence.csv": pd.DataFrame(evidence_rows),
    }
    statistical_rows = []
    for row in evidence_rows:
        p_value, alpha = row["significance_p_value"], row["alpha"]
        if (p_value is None or alpha is None
                or not np.isfinite(p_value) or not np.isfinite(alpha)):
            continue
        statistical_rows.append({
            "identity": int(identity),
            "metric": row["metric"],
            "test_name": workbench.PERIOD_METHODS[significance_method]["label"],
            "estimate_name": "best period",
            "estimate": row["period_hours"],
            "estimate_units": "h",
            "effect_size_name": "periodogram goodness of fit",
            "effect_size": row["goodness_of_fit"],
            "p_value": p_value,
            "q_value": row["q_value"],
            "correction_method": row["correction"],
            "alpha": alpha,
            "significant": bool(float(row["q_value"]) < float(alpha)),
            "n": int(cell[row["metric"]].notna().sum()),
            "detrend": detrend,
            "detrend_window_hours": detrending["detrend_window_hours"],
            "period_search_min_hours": float(rhythm_params["period_search_hours"][0]),
            "period_search_max_hours": float(rhythm_params["period_search_hours"][1]),
            "workbench_version": row["workbench_version"],
        })
    if statistical_rows:
        auxiliary["statistics.csv"] = pd.DataFrame(statistical_rows)
    if tile_frames:
        auxiliary["display_settings.csv"] = pd.DataFrame([display_settings])
        auxiliary["tile_index.csv"] = pd.DataFrame({
            "position": range(len(tile_frames)),
            "frame_index": tile_frames,
            "hours": [float(cell.loc[cell.frame_index == f, "hours"].iloc[0])
                      for f in tile_frames],
            "area_px": [float(cell.loc[cell.frame_index == f, "area_px"].iloc[0])
                        if "area_px" in cell.columns else np.nan for f in tile_frames],
            "crop_top": crop_box[0], "crop_bottom": crop_box[1],
            "crop_left": crop_box[2], "crop_right": crop_box[3],
            "display_vmin": vmin, "display_vmax": vmax,
            "lut": cell_lut or ctx.theme["image_cmap"],
            "display_only": True,
            "image_filter": display_settings["image_filter"],
            "display_gain": display_settings["time_gain"],
            "display_gamma": display_settings["gamma"],
        })

    names = ", ".join(describe(column).label.lower() for column in metrics)

    return FigureResult(
        figure=figure_,
        axes=axes,
        figure_data=figure_data,
        auxiliary=auxiliary,
        heading=f"Cell report card, identity {identity}",
        title_fields={"identity": identity, "names": names},
        subtitle=(
            f"{ctx.summary['stem']}. Present in {int(cell['frame_index'].count())} of "
            f"{ctx.summary['frames']} frames."
            + (f" Display-only presentation of the registered signal at "
               f"{len(tile_frames)} times; one crop and one range." if tile_frames else "")
        ),
        footnote=(
            f"Image filtering, gamma and ×{float(display_gain):g} temporal-difference "
            f"gain are presentation only; measurements and rhythm tests use the "
            f"unmodified table values.\nTraces: {trace_layout}, {trace_view}; "
            f"detrending: {detrend.replace('_', ' ')} "
            f"({detrending['detrend_window_hours']:g} h window); period method: "
            f"{workbench.PERIOD_METHODS[fit_method]['label']}.\n"
            f"--identity selects a cell, --metrics swaps measurements, and --fit "
            f"selects which measurements carry dashed fitted curves. "
            f"{significance_method} tests the trace, not an individual fitted component."
        ),
        readme=f"""## What the page shows

{'The cell outline over a display-only presentation of the signal at ' + str(len(tile_frames)) + ' times across the recording, then the requested measurement traces.' if tile_frames else 'The requested measurement traces.'}
Every tile shares one crop and one display range. The presentation pixels are
never passed back into measurement or rhythm analysis.

Image lookup table: {cell_lut or ctx.theme['image_cmap']}.
Image filter: {display_settings.get('image_filter', 'not drawn')}.
Temporal-difference display gain: {float(display_gain):g}.
Display gamma: {float(display_gamma):g}.
Display range: {float(display_black):g}th to {float(display_white):g}th percentile of {display_settings.get('display_range_scope', 'all frames')}.

Traces drawn: {', '.join(metrics)}.
Trace layout: {trace_layout}; trace values: {trace_view}.
Detrending: {detrend}, {detrending['detrend_window_hours']:g} h window, through Circadian Workbench {workbench.WORKBENCH_VERSION}.
{'Fitted curves on: ' + ', '.join(fitted_metrics) + ', at periods estimated by ' + workbench.PERIOD_METHODS[fit_method]['label'] + '.' if fitted_metrics else 'No fitted curve was drawn.'}
FFT-NLLS draws its returned multi-component model. Other estimators draw a
descriptive cosine at the estimated period. {significance_method} supplies the
trace-level significance test; it does not prove an individual fitted component.
Relative amplitude error is fit uncertainty, not a p-value.

## Which cell

Identity {identity}. This is a layout for any identity, not a selected result;
every cell in `cell_frame.csv` can be drawn the same way, with any of its
numeric columns as a trace.

## How it is built

The generic panels are `panels.image_strip`, `panels.trace`,
`panels.trace_overlay`, and `panels.harmonic_curve`. The display-only intensity
preparation is `panels.intensity.presentation_stack`, which calls the same
Auto-Organotypic implementation used for stills and movies. A metric's colour
and wording come from `_metrics`, so choosing another column needs no builder
change.""",
        console=(f"cell {identity}  frames {len(cell)}  metrics {metrics}  "
                 f"tiles {len(tile_frames)}  traces {trace_layout}/{trace_view}"
                 + (f"  {fit_method} fit on {fitted_metrics}" if fitted_metrics else "")),
    )


if __name__ == "__main__":
    run_figure("cell-report-card", DEFAULT_RUN)
