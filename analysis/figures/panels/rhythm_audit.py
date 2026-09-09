"""Reusable renderer for filter, detrending, and period-method audits."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.font_manager import weight_dict
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from ._contract import PanelResult
except ImportError:  # The copied standalone renderer has no package parent.
    @dataclass(frozen=True)
    class PanelResult:  # type: ignore[no-redef]
        data: pd.DataFrame
        axes: Any = None
        extra: dict[str, Any] = field(default_factory=dict)


def _bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _finish_axis(ax: Any, config: dict[str, Any]) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(config["stroke"]["axis"])
    ax.tick_params(
        width=config["stroke"]["axis"],
        length=config["stroke"]["tick_length"],
        labelsize=config["font"]["tick"],
    )


def _time_ticks(low: float, high: float, requested: float | None) -> np.ndarray:
    span = max(0.0, float(high) - float(low))
    if requested is None:
        candidates = np.asarray(
            [0.5, 1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 96, 120], dtype=float
        )
        target = span / 6.0 if span > 0 else 1.0
        step = float(candidates[np.abs(candidates - target).argmin()])
    else:
        step = float(requested)
    first = np.ceil(low / step) * step
    return np.arange(first, high + step * 0.25, step)


def _row_limit(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return 1.0
    low, high = np.nanpercentile(finite, [1, 99])
    limit = max(abs(float(low)), abs(float(high)))
    return 1.08 * limit if limit > 0 else 1.0


def _result(
    results: pd.DataFrame,
    trace_id: str,
    detrend: str,
    estimator: str,
    preprocessor: str,
) -> pd.Series | None:
    rows = results.loc[
        results["trace_id"].eq(trace_id)
        & results["detrend"].eq(detrend)
        & results["estimator"].eq(estimator)
        & results["preprocessor"].eq(preprocessor)
    ]
    return None if rows.empty else rows.iloc[0]


def _number(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _significance_text(record: pd.Series, config: dict[str, Any]) -> str:
    period = _number(record.get("estimated_period_hours"))
    if str(record.get("estimate_status")) != "ok" or not np.isfinite(period):
        return "estimate unavailable · not tested"
    correction = str(record.get("correction_method") or "none")
    evidence = _number(record.get("q_value" if correction != "none" else "p_value"))
    evidence_name = "q" if correction != "none" else "p"
    evidence_text = f"{evidence_name}={evidence:.2g}" if np.isfinite(evidence) else f"{evidence_name} unavailable"
    status = str(record.get("rhythm_status") or "not tested")
    suffix = " †" if _bool(record.get("period_underdetermined")) else ""
    return f"{period:.2f} h · {evidence_text} · {status}{suffix}"


def _fit_annotation(
    results: pd.DataFrame,
    trace_id: str,
    detrend: str,
    estimator: str,
    config: dict[str, Any],
) -> str:
    records = [
        _result(results, trace_id, detrend, estimator, preprocessor)
        for preprocessor in config["preprocessors"]
    ]
    available = [record for record in records if record is not None]
    test = (
        str(available[0].get("significance_method_label"))
        if available else "No significance test"
    )
    lines = [f"Test: {test}"]
    for preprocessor, record in zip(config["preprocessors"], records):
        label = config["preprocessor_labels"][preprocessor]
        lines.append(
            f"{label}: {_significance_text(record, config)}"
            if record is not None else f"{label}: not tested"
        )
    return "\n".join(lines)


def _plot_fit(
    ax: Any,
    traces: pd.DataFrame,
    results: pd.DataFrame,
    trace_id: str,
    detrend: str,
    estimator: str,
    limit: float,
    low_hour: float,
    high_hour: float,
    config: dict[str, Any],
) -> None:
    for preprocessor in config["preprocessors"]:
        panel = traces.loc[
            traces["trace_id"].eq(trace_id)
            & traces["display_panel"].eq("fit")
            & traces["detrend"].eq(detrend)
            & traces["estimator"].eq(estimator)
            & traces["preprocessor"].eq(preprocessor)
        ].sort_values("hours")
        if panel.empty:
            continue
        colour = config["colours"][preprocessor]
        linestyle = "-" if preprocessor == "raw" else (0, (5, 2))
        ax.plot(
            panel["hours"], panel["plotted_value"],
            color=colour, linewidth=config["stroke"]["data"], alpha=0.34,
        )
        ax.plot(
            panel["hours"], panel["fitted_value"],
            color=colour, linewidth=config["stroke"]["fit"], linestyle=linestyle,
        )
    ax.axhline(
        0, color=config["colours"]["reference"],
        linewidth=config["stroke"]["guide"], zorder=0,
    )
    ax.set_xlim(low_hour, high_hour)
    ax.set_ylim(-limit, limit)
    ax.text(
        0.02, 0.97,
        _fit_annotation(results, trace_id, detrend, estimator, config),
        transform=ax.transAxes, ha="left", va="top",
        fontsize=config["font"]["annotation"],
        color=config["colours"]["ink"],
        bbox={
            "facecolor": config["colours"]["page"],
            "edgecolor": "none", "alpha": 0.82, "pad": 1.2,
        },
    )
    _finish_axis(ax, config)


def draw_workbench_comparison(ax: Any, definition: dict[str, Any]) -> None:
    """Matplotlib adapter for a completed Workbench grid; no science or defaults.

    Columns, formatting, units, rows and style are supplied by the shared figure
    builder. Only placement within Motion's surrounding panel belongs here.
    """
    if definition.get("kind") != "grid":
        raise ValueError("A Workbench comparison grid is required")
    columns = definition["geometry"]["columns"]
    rows = definition["geometry"]["rows"]
    display = definition["options"]["display"]
    style = display["theme_values"]
    ax.set_axis_off()
    cells = []
    for row in rows:
        rendered = []
        for column in columns:
            key, template = column["key"], column["format"]
            value = row.get(key)
            text = ("" if value is None else
                    "yes" if key == "significant" and value else
                    "no" if key == "significant" else
                    template.format(float(value)) if template else str(value))
            if key == "label" and row.get("period_hours") is None:
                text += " — " + str(row.get("status") or "no result")
            rendered.append(text)
        cells.append(rendered)
    if not cells:
        ax.text(0, 0.5, "No completed comparison", transform=ax.transAxes)
        return
    total_width = sum(column["width"] for column in columns)
    table = ax.table(cellText=cells, colLabels=[c["label"] for c in columns],
                     colWidths=[c["width"] / total_width for c in columns],
                     cellLoc="right", loc="center", bbox=[0, 0.28, 1, 0.66])
    table.auto_set_font_size(False)
    # Figure definitions express browser/export font sizes in pixels. Convert
    # physical 96-dpi pixels to Matplotlib points without changing text scale.
    for (row_index, column_index), cell in table.get_celld().items():
        text = cell.get_text()
        text.set_fontfamily(style["font_name"])
        text.set_fontsize(float(style["axis_size"] if row_index == 0 else style["tick_size"]) * 0.75)
        text.set_color(style["text_fill"] if row_index == 0 else style["tick_fill"])
        weight = style["axis_weight"] if row_index == 0 else style["tick_weight"]
        # Matplotlib's editable-text SVG adapter requires a named weight even
        # though its raster adapter accepts the shared CSS numeric weight.
        if isinstance(weight, (int, float)):
            weight = next(name for name, number in weight_dict.items() if number == weight)
        text.set_fontweight(weight)
        text.set_ha("left" if column_index == 0 else "right")
        cell.set_facecolor("none")
        cell.visible_edges = "B"
        cell.set_edgecolor(style["spine_stroke"] if row_index == 0 else style["grid_stroke"])
        cell.set_linewidth(float(style["spine_width"] if row_index == 0 else style["grid_width"]) * 0.75)
    caption_size = float(style["tick_size"]) * 0.75
    panel_points = ax.get_position().width * ax.figure.get_figwidth() * 72
    caption = " ".join(definition.get("annotations", ())) + " Raw method p-values; family-corrected verdicts are shown above."
    ax.text(0, 0.04, "\n".join(textwrap.wrap(caption, max(20, int(panel_points / (caption_size * 0.55))))),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=float(style["tick_size"]) * 0.75,
            fontfamily=style["font_name"], color=style["text_fill"])


def draw(
    figure: Any,
    rect: tuple[float, float, float, float],
    traces: pd.DataFrame,
    results: pd.DataFrame,
    components: pd.DataFrame,
    spectra: pd.DataFrame,
    config: dict[str, Any],
) -> PanelResult:
    """Draw a duration-aware audit grid for every selected cell trace."""
    trace_keys = list(config["trace_keys"])
    detrends = list(config["detrends"])
    estimators = list(config["estimators"])
    include_components = "fft_nlls" in estimators
    include_spectrum = "mesa" in estimators
    column_kinds = ["detrended", *estimators]
    if include_components:
        column_kinds.append("fft_components")
    if include_spectrum:
        column_kinds.append("mesa_spectrum")
    n_columns = len(column_kinds)
    shared_comparisons = bool(config.get("shared_comparisons"))
    row_stride = 2 if shared_comparisons else 1

    height_ratios: list[float] = []
    block_starts: list[int] = []
    cursor = 0
    for index, _ in enumerate(trace_keys):
        block_starts.append(cursor)
        height_ratios.append(0.78)
        for _ in detrends:
            height_ratios.append(1.0)
            if shared_comparisons:
                height_ratios.append(config["comparison_row_height"])
        cursor += 1 + row_stride * len(detrends)
        if index < len(trace_keys) - 1:
            height_ratios.append(0.28)
            cursor += 1
    grid = GridSpec(
        len(height_ratios), n_columns, figure=figure,
        height_ratios=height_ratios,
        width_ratios=[
            1.08 if kind == "detrended" else
            0.86 if kind == "fft_components" else
            1.0 if kind == "mesa_spectrum" else 1.12
            for kind in column_kinds
        ],
        left=rect[0], bottom=rect[1], right=rect[0] + rect[2],
        top=rect[1] + rect[3], hspace=0.72, wspace=0.32,
    )
    axes: list[Any] = []

    column_titles = []
    for kind in column_kinds:
        if kind == "detrended":
            column_titles.append("Detrended data")
        elif kind == "fft_components":
            column_titles.append("Nonlinear least-squares\nreturned components")
        elif kind == "mesa_spectrum":
            column_titles.append("Maximum entropy spectral analysis\nreturned spectrum")
        else:
            label = config["estimator_labels"].get(kind, kind.replace("_", " ").title())
            suffix = (
                "returned multi-component model"
                if kind == "fft_nlls" else "descriptive cosine display fit"
            )
            column_titles.append("\n".join(textwrap.wrap(label, 28)) + "\n" + suffix)

    for block_index, trace_key in enumerate(trace_keys):
        trace_id = str(trace_key["trace_id"])
        identity = int(trace_key["identity"])
        metric = str(trace_key["metric"])
        start_row = block_starts[block_index]
        raw_axis = figure.add_subplot(grid[start_row, :])
        axes.append(raw_axis)
        raw = traces.loc[
            traces["trace_id"].eq(trace_id)
            & traces["display_panel"].eq("raw_reference")
        ].sort_values("hours")
        raw_axis.plot(
            raw["hours"], raw["plotted_value"],
            color=config["colours"]["ink"], linewidth=config["stroke"]["fit"],
        )
        raw_axis.scatter(
            raw["hours"], raw["plotted_value"],
            color=config["colours"]["ink"], s=config["point_area"], zorder=3,
        )
        low_hour = float(raw["hours"].min())
        high_hour = float(raw["hours"].max())
        ticks = _time_ticks(low_hour, high_hour, config.get("hour_ticks"))
        raw_axis.set_xlim(low_hour, high_hour)
        raw_axis.set_xticks(ticks)
        raw_axis.set_ylabel("Raw value", fontsize=config["font"]["axis"])
        raw_axis.set_title(
            f"Raw reference — cell identity {identity}, {metric}",
            loc="left", fontsize=config["font"]["row"], fontweight="bold",
        )
        _finish_axis(raw_axis, config)

        for detrend_index, detrend in enumerate(detrends):
            grid_row = start_row + 1 + row_stride * detrend_index
            detrended = traces.loc[
                traces["trace_id"].eq(trace_id)
                & traces["display_panel"].eq("detrended")
                & traces["detrend"].eq(detrend)
            ]
            limit = _row_limit(detrended["plotted_value"])
            row_axes: list[Any] = []

            data_axis = figure.add_subplot(grid[grid_row, 0])
            row_axes.append(data_axis)
            for preprocessor in config["preprocessors"]:
                panel = detrended.loc[
                    detrended["preprocessor"].eq(preprocessor)
                ].sort_values("hours")
                data_axis.plot(
                    panel["hours"], panel["plotted_value"],
                    color=config["colours"][preprocessor],
                    linewidth=config["stroke"]["line"],
                    linestyle="-" if preprocessor == "raw" else (0, (5, 2)),
                )
            data_axis.axhline(
                0, color=config["colours"]["reference"],
                linewidth=config["stroke"]["guide"], zorder=0,
            )
            data_axis.set_xlim(low_hour, high_hour)
            data_axis.set_ylim(-limit, limit)
            data_axis.set_ylabel(
                config["detrend_labels"].get(
                    detrend, detrend.replace("_", " ").capitalize()
                ) + "\nDetrended value",
                fontsize=config["font"]["row"], fontweight="bold",
            )
            _finish_axis(data_axis, config)

            column_index = 1
            for estimator in estimators:
                axis = figure.add_subplot(grid[grid_row, column_index])
                _plot_fit(
                    axis, traces, results, trace_id, detrend, estimator,
                    limit, low_hour, high_hour, config,
                )
                row_axes.append(axis)
                column_index += 1

            span = high_hour - low_hour
            cycle_ceiling = span / float(config["min_cycles"])
            if include_components:
                axis = figure.add_subplot(grid[grid_row, column_index])
                if config["period_min_hours"] < cycle_ceiling < config["period_max_hours"]:
                    axis.axvline(
                        cycle_ceiling, color=config["colours"]["warning"],
                        linewidth=config["stroke"]["guide"], linestyle=(0, (3, 2)),
                    )
                for preprocessor, y_base in zip(config["preprocessors"], [1.0, 0.0]):
                    subset = components.loc[
                        components["trace_id"].eq(trace_id)
                        & components["detrend"].eq(detrend)
                        & components["estimator"].eq("fft_nlls")
                        & components["preprocessor"].eq(preprocessor)
                    ]
                    for local_index, (_, component) in enumerate(subset.iterrows()):
                        y = y_base + 0.12 * (local_index - (len(subset) - 1) / 2)
                        period = _number(component.get("period_hours"))
                        error = _number(component.get("period_error_hours"))
                        selected = _bool(component.get("selected"))
                        if not np.isfinite(period):
                            continue
                        colour = config["colours"][preprocessor]
                        axis.errorbar(
                            period, y, xerr=error if np.isfinite(error) else None,
                            fmt="o" if selected else "D", markersize=config["component_marker"],
                            markerfacecolor=colour if selected else config["colours"]["page"],
                            markeredgecolor=colour, color=colour,
                            linewidth=config["stroke"]["guide"], capsize=1.7,
                        )
                        axis.text(
                            period, y + 0.13, f"{period:.1f}", ha="center", va="bottom",
                            fontsize=config["font"]["small"], clip_on=True,
                        )
                axis.set_xlim(config["period_min_hours"], config["period_max_hours"])
                axis.set_ylim(-0.45, 1.45)
                axis.set_yticks([0, 1], ["Median", "Raw"])
                axis.text(
                    0.02, 0.03, "Component significance not tested",
                    transform=axis.transAxes, ha="left", va="bottom",
                    fontsize=config["font"]["small"],
                )
                if config["period_min_hours"] < cycle_ceiling < config["period_max_hours"]:
                    axis.text(
                        cycle_ceiling + 0.01 * (config["period_max_hours"] - config["period_min_hours"]),
                        1.38, "cycle ceiling", color=config["colours"]["warning"],
                        ha="left", va="top", fontsize=config["font"]["small"],
                    )
                _finish_axis(axis, config)
                row_axes.append(axis)
                column_index += 1

            if include_spectrum:
                axis = figure.add_subplot(grid[grid_row, column_index])
                if config["period_min_hours"] < cycle_ceiling < config["period_max_hours"]:
                    axis.axvline(
                        cycle_ceiling, color=config["colours"]["warning"],
                        linewidth=config["stroke"]["guide"], linestyle=(0, (3, 2)),
                    )
                peak_lines = []
                for preprocessor in config["preprocessors"]:
                    spectrum = spectra.loc[
                        spectra["trace_id"].eq(trace_id)
                        & spectra["detrend"].eq(detrend)
                        & spectra["estimator"].eq("mesa")
                        & spectra["preprocessor"].eq(preprocessor)
                    ].sort_values("period_hours")
                    colour = config["colours"][preprocessor]
                    axis.plot(
                        spectrum["period_hours"], spectrum["normalised_power"],
                        color=colour, linewidth=config["stroke"]["line"],
                        linestyle="-" if preprocessor == "raw" else (0, (5, 2)),
                    )
                    record = _result(results, trace_id, detrend, "mesa", preprocessor)
                    if record is not None:
                        period = _number(record.get("estimated_period_hours"))
                        if np.isfinite(period):
                            axis.axvline(
                                period, color=colour, linewidth=config["stroke"]["guide"],
                                linestyle="-" if preprocessor == "raw" else (0, (5, 2)),
                                alpha=0.75,
                            )
                        peaks = _number(record.get("mesa_peaks_in_search_band"))
                        peak_lines.append(
                            f"{config['preprocessor_labels'][preprocessor]}: "
                            + (f"{int(peaks)}" if np.isfinite(peaks) else "unavailable")
                        )
                axis.set_xlim(config["period_min_hours"], config["period_max_hours"])
                axis.set_ylim(0, 1.08)
                axis.text(
                    0.02, 0.97,
                    "Workbench peaks in band\n" + " · ".join(peak_lines)
                    + "\nPeak significance not tested",
                    transform=axis.transAxes, ha="left", va="top",
                    fontsize=config["font"]["small"],
                    bbox={
                        "facecolor": config["colours"]["page"],
                        "edgecolor": "none", "alpha": 0.82, "pad": 1.2,
                    },
                )
                _finish_axis(axis, config)
                row_axes.append(axis)

            axes.extend(row_axes)
            if detrend_index == 0:
                for axis, title in zip(row_axes, column_titles):
                    axis.set_title(
                        title, fontsize=config["font"]["column"],
                        fontweight="bold", pad=7,
                    )
            if detrend_index < len(detrends) - 1:
                for axis in row_axes:
                    axis.set_xticklabels([])
            else:
                row_axes[0].set_xlabel("Hours from start of recording", fontsize=config["font"]["axis"])
                for estimator_axis in row_axes[1: 1 + len(estimators)]:
                    estimator_axis.set_xlabel("Hours from start of recording", fontsize=config["font"]["axis"])
                index = 1 + len(estimators)
                if include_components:
                    row_axes[index].set_xlabel("Period (h)", fontsize=config["font"]["axis"])
                    index += 1
                if include_spectrum:
                    row_axes[index].set_xlabel("Period (h)", fontsize=config["font"]["axis"])
            for axis in row_axes[: 1 + len(estimators)]:
                axis.set_xticks(ticks)

            if shared_comparisons:
                tables = grid[grid_row + 1, :].subgridspec(1, len(config["preprocessors"]), wspace=0.12)
                for index, preprocessor in enumerate(config["preprocessors"]):
                    axis = figure.add_subplot(tables[0, index])
                    axes.append(axis)
                    entries = results.loc[
                        results["trace_id"].eq(trace_id) & results["detrend"].eq(detrend)
                        & results["preprocessor"].eq(preprocessor)]
                    value = entries.iloc[0].get("workbench_comparison_json") if not entries.empty else None
                    if isinstance(value, str) and value:
                        draw_workbench_comparison(axis, json.loads(value))
                    else:
                        axis.set_axis_off()
                        axis.text(0, 0.5, "Comparison unavailable; see method refusal above.", transform=axis.transAxes)
                    axis.set_title(config["preprocessor_labels"][preprocessor] + " — Workbench comparison",
                                   loc="left", fontsize=config["font"]["column"], fontweight="bold")

    handles = [
        Line2D(
            [0], [0], color=config["colours"][preprocessor],
            linewidth=config["stroke"]["fit"],
            linestyle="-" if preprocessor == "raw" else (0, (5, 2)),
            label=config["preprocessor_labels"][preprocessor],
        )
        for preprocessor in config["preprocessors"]
    ]
    figure.legend(
        handles=handles, loc="upper right",
        bbox_to_anchor=(rect[0] + rect[2], config["legend_y"]),
        frameon=False, ncol=len(handles), fontsize=config["font"]["legend"],
    )
    figure.text(
        config["header_x"], config["title_y"], config["title"],
        ha="left", va="top", fontsize=config["font"]["title"],
        fontweight="bold", color=config["colours"]["ink"],
    )
    figure.text(
        config["header_x"], config["subtitle_y"], config["subtitle"],
        ha="left", va="top", fontsize=config["font"]["subtitle"],
        color=config["colours"]["caption"],
    )
    figure.text(
        config["header_x"], config["footnote_y"], config["footnote"],
        ha="left", va="bottom", fontsize=config["font"]["footnote"],
        color=config["colours"]["caption"], wrap=True,
    )
    return PanelResult(data=traces.copy(), axes=axes)
