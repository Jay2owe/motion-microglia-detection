"""Shared options, data preparation and exact producers for spatial figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import tifffile

from analysis import spatial as measurements
from analysis import circadian as workbench
from analysis.modules.rhythms import DEFAULTS as RHYTHM_DEFAULTS
from _metrics import semantic_label
from _schema import Option, Table, Input, FigureResult, _placement
from _options import Length
from _text import figure_text
from panels import spatial


READS = (Table("cell_frame.csv", module="measurement modules"),
         Table("sholl.csv", module="sholl", optional=True), Input("labels"))
COMMON_OPTIONS = (
    Option("spatial_metric", default="radial_occupancy"),
    Option("display", default="standardized"), Option("detrend", default=None),
    Option("detrend_window_hours", default=None),
    Option("spatial_centre", default="soma"), Option("spatial_axis", default="y"),
    Option("spatial_panel_inches", default=6.), Option("spatial_point_size", default=150.),
    Option("spatial_annotate", default=False), Option("spatial_cmap", default=None),
    Option("snapshot_hours", default=[]), Option("snapshot_count", default=1),
    Option("show_history", default=False), Option("spatial_arrows", default=False),
    Option("spatial_tracks", default=True), Option("spatial_track_cells", default="rhythmic"),
    Option("spatial_track_width", default=1.6),
    Option("phase_group_hours", default=[]), Option("period_tolerance", default=.1),
    Option("timing_reference_hour", default=None), Option("hour_ticks", default=None),
    Option("annulus_support", default=.5), Option("scaling", default="cell"),
    Option("max_inferred_fraction", default=1.),
)
PERIOD_OPTIONS = (
    Option("fit_method", default=None), Option("significance_method", default=None),
    Option("period_config", default={}), Option("period_min_hours", default=2.),
    Option("period_max_hours", default=48.), Option("multiple_testing", default="bh"),
    Option("rhythmic_alpha", default=.05), Option("min_cycles", default=3.),
    Option("min_observations", default=24),
)


def options_for(kind):
    shared = {"spatial_metric", "display", "detrend", "detrend_window_hours", "spatial_centre",
              "spatial_panel_inches", "spatial_point_size", "spatial_annotate", "spatial_cmap",
              "hour_ticks", "annulus_support", "scaling", "max_inferred_fraction"}
    extras = {
        "expansion": {"snapshot_hours", "snapshot_count", "spatial_arrows"},
        "period": {"phase_group_hours", "period_tolerance", "spatial_tracks", "spatial_track_cells", "spatial_track_width"},
        "progression": {"spatial_axis", "snapshot_hours", "snapshot_count", "show_history"},
        "timing": {"period_tolerance", "timing_reference_hour"},
        "neighbours": set(), "gaps": {"snapshot_hours", "snapshot_count", "show_history"},
    }
    if kind == "gaps":
        shared = {"spatial_panel_inches", "spatial_cmap", "hour_ticks"}
    if kind in {"period", "timing"}:
        shared -= {"display", "hour_ticks"}
    if kind in {"expansion", "neighbours"}:
        shared -= {"spatial_point_size", "spatial_annotate"}
    common = tuple(opt for opt in COMMON_OPTIONS if opt.name in shared | extras[kind])
    extra = ()
    if kind in {"period", "timing"}:
        extra = PERIOD_OPTIONS
    if kind == "timing":
        extra += (Option("reference_metric", default="corrected_mean"),)
    if kind == "neighbours":
        extra = (Option("spatial_neighbours", default=3), Option("spatial_permutations", default=999),
                 Option("spatial_seed", default=20260907), Option("min_observations", default=24))
    return common + extra


def _get(ctx, name):
    """Internal defaults for unused shared plumbing are not advertised flags."""
    if name in {option.name for option in ctx.spec.options}:
        return ctx.option(name)
    defaults = {option.name: option.default for option in COMMON_OPTIONS}
    return defaults[name]


def _metric_name(name):
    return measurements.RADIAL_METRIC if name == "radial_occupancy" else str(name)


def _label(metric):
    if metric == measurements.RADIAL_METRIC:
        return "Mean relative radial occupancy"
    try:
        return semantic_label(metric)
    except ValueError:
        return metric.replace("_", " ")


def _source_data(ctx, metrics):
    frame = ctx.table("cell_frame.csv")
    if measurements.RADIAL_METRIC in metrics:
        sholl = ctx.table("sholl.csv")
        reduced = measurements.radial_occupancy(
            sholl, scaling=str(_get(ctx, "scaling")), support=float(_get(ctx, "annulus_support")),
            length_per_pixel=float(ctx.scale.microns_per_pixel) if ctx.scale.calibrated else 1.)
        frame = frame.merge(reduced.drop(columns="hours"), on=["identity", "frame_index"],
                            how="left", validate="one_to_one")
    for name in metrics:
        if name not in frame or not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"spatial metric {name!r} is not a numeric column")
    maximum = float(_get(ctx, "max_inferred_fraction"))
    if not 0 <= maximum <= 1:
        raise ValueError("max_inferred_fraction must be between zero and one")
    if maximum < 1:
        if "inferred_fraction" not in frame:
            raise ValueError("this run has no reconstruction fraction for the requested filter")
        frame.loc[frame.inferred_fraction.isna() | frame.inferred_fraction.gt(maximum), metrics] = np.nan
    params = {**RHYTHM_DEFAULTS, **ctx.module_params("rhythms")}
    params.update(workbench.detrend_settings(params))
    return frame, params


def _compact_fits(fits):
    # Spectral arrays are not displayed. The complete estimator metadata is
    # retained as JSON, rather than unstable Python repr strings in CSV cells.
    out = fits.copy()
    for name in ("components", "estimate_diagnostics"):
        if name in out:
            out[name] = out[name].map(lambda obj: json.dumps(obj, default=str))
    return out


def _fits(ctx, frame, metrics, params):
    resolved = workbench.resolve_analysis_options(params, ctx.option)
    params = resolved["params"]
    extra = params["workbench_config"]
    if extra.get("phase_reference", "zero") != "zero":
        raise ValueError("spatial timing requires phase_reference=zero (one shared recording origin)")
    params["workbench_config"] = {**extra, "phase_reference": "zero"}
    fits = measurements.rhythm_fits(
        frame, metrics, params, method=resolved["method"],
        significance_method=resolved["significance_method"],
        correction=resolved["multiple_testing"],
        min_observations=resolved["min_observations"],
        min_cycles=resolved["min_cycles"])
    return fits, params


def _pixels(labels, frame, indices, hours):
    rows = []
    for tile, index in enumerate(indices):
        yy, xx = np.nonzero(labels[index])
        ids = labels[index, yy, xx].astype(int)
        selected = frame[frame.frame_index.eq(index)].set_index("identity")
        values = selected.value.reindex(ids).to_numpy(float)
        rows.append(pd.DataFrame(dict(record="pixel", tile=tile, hours=hours[index],
                                      x=xx, y=yy, identity=ids, value=values)))
    return pd.concat(rows, ignore_index=True)


def _period_data(ctx, fits, pos):
    cells = pos.merge(fits, on="identity", how="left", validate="one_to_one")
    cells["record"] = "cell"
    cells["tile"] = "period"
    cells["value"] = cells.period_hours.where(cells.significant & cells.period_available)
    low, high = map(float, fits[["period_search_min_hours", "period_search_max_hours"]].iloc[0])
    tiles = [dict(key="period", title="Detected period", label="Estimated period (h)",
                  cmap=_get(ctx, "spatial_cmap") or "viridis", vmin=low, vmax=high)]
    groups = _get(ctx, "phase_group_hours")
    tolerance = float(_get(ctx, "period_tolerance"))
    if not 0 <= tolerance < 1:
        raise ValueError("period_tolerance must be in [0, 1)")
    if not groups:
        keep = (cells.significant.fillna(False).astype(bool)
                & cells.period_available.fillna(False).astype(bool)
                & cells.period_hours.gt(0) & cells.phase_hours.notna())
        supported = keep & cells.supported.fillna(False).astype(bool)
        phase = cells.copy()
        phase["tile"] = "phase_percent"
        phase["phase_fraction"] = (
            phase.phase_hours.mod(phase.period_hours) / phase.period_hours
        ).where(keep)
        phase["value"] = phase["phase_fraction"] * 100
        phase.loc[~keep, "display_status"] = "unsupported phase"
        phase.loc[keep & ~supported, "display_status"] = "exploratory period"
        phase.loc[supported, "display_status"] = "supported rhythm"
        output = [cells, phase]
        tiles.append(dict(
            key="phase_percent",
            title=(f"Peak within each cell's own cycle\n{int(keep.sum())} significant; "
                   f"{int(supported.sum())} supported"),
            label="Peak position within own cycle (%)",
            cmap="phase_cycle", vmin=0, vmax=100,
        ))
        return pd.concat(output, ignore_index=True), tiles
    output = [cells]
    for i, centre in enumerate(groups):
        centre = float(centre)
        if not np.isfinite(centre) or centre <= 0:
            raise ValueError("phase-group periods must be finite and positive")
        keep = cells.supported & ((cells.period_hours - centre).abs() / centre <= tolerance)
        group = cells.copy()
        group["tile"] = f"phase_{i}"
        group["value"] = (group.phase_hours.mod(group.period_hours) / group.period_hours * centre).where(keep)
        group["group_period_hours"] = centre
        group.loc[~keep, "display_status"] = "excluded from period group"
        output.append(group)
        tiles.append(dict(key=f"phase_{i}", title=f"Peak timing: {centre:.2g} h group\n{int(keep.sum())} supported cells",
                          label=f"Peak phase (h on a {centre:.2g} h group cycle)",
                          cmap="phase_cycle", vmin=0, vmax=centre))
    return pd.concat(output, ignore_index=True), tiles


def period_track_data(frame, data, *, scope="rhythmic", centre="soma", scale=1.):
    """Colour actual paths by the saved period, selecting by the saved test verdict."""
    if scope not in {"rhythmic", "all"}:
        raise ValueError("spatial_track_cells must be rhythmic or all")
    cells = data[data.record.eq("cell") & data.tile.eq("period")]
    if scope == "rhythmic":
        if "significant" not in cells:
            raise ValueError("rhythmic-only tracks require saved significance verdicts")
        cells = cells[cells.significant.astype(str).str.lower().isin(["true", "1", "1.0"])]
    segments = measurements.track_segments(frame, centre=centre, length_per_pixel=scale)
    return segments.merge(cells[["identity", "value", "significant", "display_status"]],
                          on="identity", how="inner", validate="many_to_one").assign(record="track", tile="period")


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build(ctx, kind):
    # A single panel declaration can describe a generic multi-map block. Reject
    # empty selections instead of drawing a plot whose panel was disabled.
    if not ctx.panels():
        raise ValueError("select the spatial plot panel")
    metric = _metric_name(_get(ctx, "spatial_metric"))
    metrics = [metric]
    if kind == "timing":
        reference = _metric_name(_get(ctx, "reference_metric"))
        if reference == metric:
            raise ValueError("choose different metrics for the timing comparison")
        metrics.append(reference)
    frame, params = _source_data(ctx, [] if kind == "gaps" else metrics)
    scale = float(ctx.scale.microns_per_pixel) if ctx.scale.calibrated else 1.
    centre = str(_get(ctx, "spatial_centre"))
    pos = (measurements.positions(frame, centre=centre, length_per_pixel=scale)
           if kind != "gaps" else pd.DataFrame(columns=["identity", "x", "y"]))
    hour_rows = frame.groupby("frame_index").hours.agg(["min", "max"])
    if not hour_rows["min"].eq(hour_rows["max"]).all():
        raise ValueError("cells disagree on the time of a frame")
    hours = hour_rows["min"].to_numpy(float)
    full_indices = np.arange(int(ctx.field["frames"]))
    if not np.array_equal(hour_rows.index, full_indices):
        raise ValueError("a complete frame-time axis is required; an empty frame needs explicit timing")
    if len(hours) < 2:
        raise ValueError("spatial time-series figures need at least two frames")
    size = float(_get(ctx, "spatial_panel_inches"))
    if not np.isfinite(size) or size < 5.5:
        raise ValueError("spatial panels must be at least 5.5 inches wide")
    point = float(_get(ctx, "spatial_point_size"))
    if not np.isfinite(point) or point <= 0:
        raise ValueError("spatial_point_size must be finite and positive")
    cfg = dict(kind=kind, shape=[int(ctx.field["height"]), int(ctx.field["width"])],
               scale=scale, length_unit=ctx.length_label, panel_inches=size,
               point_size=point, annotate=bool(_get(ctx, "spatial_annotate")),
               hours=hours.tolist(), recording_hours=hours.tolist(), hour_ticks=list(ctx.theme.hour_ticks(float(hours.min()), float(hours.max()),
                                                                        _get(ctx, "hour_ticks"))),
               line_color=ctx.theme.colour("reporter"))
    auxiliary = {"input_traces.csv": frame[["identity", "frame_index", "hours", *([] if kind == "gaps" else metrics)]].copy(),
                 "positions.csv": pos, **ctx.provenance_auxiliary()}
    label = _label(metric)
    display = str(_get(ctx, "display"))
    title = ctx.spec.title
    limits = (-2., 2.)
    foot = "Positions and occupancy come from tracked cell labels; reconstruction can influence these maps. "
    if kind in {"expansion", "progression", "gaps"}:
        indices = measurements.snapshot_indices(hours, _get(ctx, "snapshot_hours"),
                                                count=_get(ctx, "snapshot_count"))
        cfg["snapshots"] = [dict(tile=tile, frame_index=int(index), hours=float(hours[index]))
                            for tile, index in enumerate(indices)]
        cfg["show_history"] = bool(_get(ctx, "show_history"))
        cfg["snapshot_count"] = _get(ctx, "snapshot_count")
        cfg["snapshot_hours"] = _get(ctx, "snapshot_hours")
        auxiliary["snapshots.csv"] = pd.DataFrame(cfg["snapshots"])
        selected_label = (f"Final recorded frame: {hours[-1]:g} h." if list(indices) == [len(hours) - 1]
                          else "Selected recorded hours: " + ", ".join(f"{hours[i]:g}" for i in indices) + ".")
        foot += "Missing cells/values remain missing; earlier observations are not carried forward. "
    if metric == measurements.RADIAL_METRIC and kind != "gaps":
        foot += "Relative radial occupancy measures filling of the current reach, not absolute process extension. "
    if kind in {"expansion", "progression", "neighbours"}:
        traced = measurements.display_traces(frame, metric, params, display)
        finite = traced.value.to_numpy(float)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            raise ValueError("no finite display values remain after metric selection and detrending")
        if display == "raw":
            limits = (float(np.min(finite)), float(np.max(finite)))
        else:
            limit = max(float(np.percentile(np.abs(finite), 98)), 1e-9)
            limits = (-limit, limit)
        cfg["value_label"] = label + ({"raw": " (raw)", "residual": " (baseline removed)",
                                        "standardized": " (within-cell standard deviations)"}[display])
        auxiliary["display_traces.csv"] = traced[["identity", "frame_index", "hours", "value"]]
        foot += f"Display: {display}; baseline: {params['detrend']}. Missing observations are not interpolated. "
    if kind in {"expansion", "gaps"}:
        source = ctx.input_path("labels")
        if source is None:
            raise ValueError("this plot needs the label movie recorded in the run manifest")
        labels = tifffile.imread(source)
        if labels.shape != (len(hours), *cfg["shape"]):
            raise ValueError("the label movie does not match the analysed frames and field")
    if kind == "expansion":
        data = _pixels(labels, traced, indices, hours)
        if bool(_get(ctx, "spatial_arrows")):
            ordered = traced.sort_values(["identity", "frame_index"])
            prior = ordered.groupby("identity").shift()
            arrows = []
            for tile, index in enumerate(indices):
                valid = ordered.frame_index.eq(index) & (ordered.frame_index - prior.frame_index).eq(1)
                current = ordered[valid]
                arrows.append(pd.DataFrame(dict(record="arrow", tile=tile, hours=hours[index],
                    identity=current.identity, x=prior.loc[valid, f"{centre}_x"] * scale,
                    y=prior.loc[valid, f"{centre}_y"] * scale,
                    x2=current[f"{centre}_x"] * scale, y2=current[f"{centre}_y"] * scale)))
            data = pd.concat([data, *arrows], ignore_index=True)
            foot += f"Arrows show {centre} displacement since the preceding frame, at true spatial scale. "
        subtitle = (f"{label}; " +
                    ("raw measurements." if display == "raw" else "each cell compared with its own baseline."))
    elif kind == "progression":
        axis = str(_get(ctx, "spatial_axis"))
        if axis not in {"x", "y"}:
            raise ValueError("spatial_axis must be x or y")
        cells = pos.dropna(subset=["x", "y"]).sort_values([axis, "identity"]).copy()
        cells["spatial_order"] = np.arange(1, len(cells) + 1)
        cells["record"] = "cell"
        shown = traced if cfg["show_history"] else traced[traced.frame_index.isin(indices)]
        cfg["hours"] = hours.tolist() if cfg["show_history"] else hours[indices].tolist()
        data = pd.concat([cells, shown[["identity", "hours", "value"]].assign(record="trace")], ignore_index=True)
        cfg["spatial_axis"] = axis
        subtitle = f"{label}; cells ordered by median {centre} {axis.upper()} position, not by their peak times."
        if cfg["show_history"]:
            foot += "Repeated diagonal bands indicate sequential timing, not proof of a propagating signal."
    elif kind == "period":
        fits, params = _fits(ctx, frame, metrics, params)
        data, cfg["tiles"] = _period_data(ctx, fits, pos)
        if not isinstance(_get(ctx, "spatial_tracks"), bool):
            raise ValueError("spatial_tracks must be true or false")
        cfg.update(spatial_tracks=_get(ctx, "spatial_tracks"),
                   spatial_track_cells=str(_get(ctx, "spatial_track_cells")),
                   spatial_track_width=float(_get(ctx, "spatial_track_width")))
        if cfg["spatial_track_cells"] not in {"rhythmic", "all"}:
            raise ValueError("spatial_track_cells must be rhythmic or all")
        if not np.isfinite(cfg["spatial_track_width"]) or cfg["spatial_track_width"] <= 0:
            raise ValueError("spatial_track_width must be finite and positive")
        if cfg["spatial_tracks"]:
            tracks = period_track_data(frame, data, scope=cfg["spatial_track_cells"], centre=centre, scale=scale)
            data = pd.concat([data, tracks], ignore_index=True)
            auxiliary["tracks.csv"] = tracks
            cfg["track_note"] = (f"Tracks: {cfg['spatial_track_cells']} cells, recorded {centre} paths; gaps are not connected. "
                                 "Track colour is the detected period; grey means no displayed period. ")
            foot += cfg["track_note"]
        auxiliary["statistics.csv"] = _compact_fits(fits)
        subtitle = f"{label}; {int(fits.significant.sum())} significant traces, {int(fits.supported.sum())} with supported periods."
        foot += "Period and peak maps retain all significant estimates with available periods and phases. "
        if _get(ctx, "phase_group_hours"):
            foot += "Peak phase is expressed in hours on each named group's cycle, not on a universal 24 h clock. "
        else:
            foot += ("Peak position is expressed as a percentage of each significant cell's own estimated cycle. "
                     "Outlined peaks have exploratory periods that do not meet the support requirements. "
                     "This normalization does not establish comparable timing, synchrony or a shared tissue clock. ")
    elif kind == "timing":
        fits, params = _fits(ctx, frame, metrics, params)
        when = _get(ctx, "timing_reference_hour")
        when = float((hours[0] + hours[-1]) / 2 if when is None else when)
        if not np.isfinite(when) or not hours[0] <= when <= hours[-1]:
            raise ValueError("timing_reference_hour must lie within the recording")
        data = measurements.timing_difference(fits, metric, reference,
                    period_tolerance=float(_get(ctx, "period_tolerance")), reference_hour=when)
        data = pos.merge(data, on="identity", how="left")
        auxiliary["statistics.csv"] = _compact_fits(fits)
        finite = data.value.dropna()
        limit = max(float(finite.abs().max()), 1) if len(finite) else 1
        limits = (-limit, limit)
        subtitle = f"{label} relative to {_label(reference)} at recording hour {when:g}; {int(data.supported.sum())} comparable cells."
        foot += "Positive delay means the selected metric peaks later. Only supported, similar-period rhythms are compared. "
        foot += "Different periods imply changing offsets; the data table reports the predicted drift. "
    elif kind == "neighbours":
        pairs, null, statistics = measurements.neighbour_coordination(
            traced, pos, neighbours=int(_get(ctx, "spatial_neighbours")),
            min_overlap=int(_get(ctx, "min_observations")), permutations=int(_get(ctx, "spatial_permutations")),
            seed=int(_get(ctx, "spatial_seed")))
        finite = null.value.dropna().to_numpy(float)
        counts, edges = np.histogram(finite, bins=30)
        bins = pd.DataFrame(dict(record="null_bin", x=edges[:-1], x2=edges[1:], value=counts))
        data = pd.concat([pairs.assign(record="pair"), pos.assign(record="cell"), bins], ignore_index=True)
        auxiliary.update({"null_distribution.csv": null, "statistics.csv": statistics})
        p = float(statistics.p_value.iloc[0])
        cfg["test_label"] = f"Whole-trace spatial shuffle\ntwo-sided p = {p:.3g}" if np.isfinite(p) else "Spatial comparison unavailable"
        cfg["contrast"] = _safe(float(statistics.estimate.iloc[0]))
        subtitle = f"{label}; {_get(ctx, 'spatial_neighbours')} nearest neighbours per cell, using median {centre} positions."
        foot += "Edges are proximity, not measured contact. Complete traces are shuffled within coverage strata. "
        foot += "One field-level association test; neither cell pairs nor pixels are independent experimental replicates."
    elif kind == "gaps":
        domain, age, coverage = measurements.coverage_gaps(labels > 0, hours)
        yy, xx = np.nonzero(domain)
        if not len(xx):
            raise ValueError("no occupied tissue in this label movie")
        data = pd.concat([pd.DataFrame(dict(record="pixel", tile=tile, hours=hours[index],
                                            x=xx, y=yy, value=age[index, yy, xx]))
                          for tile, index in enumerate(indices)], ignore_index=True)
        shown_coverage = coverage if cfg["show_history"] else coverage.iloc[indices]
        data = pd.concat([data, shown_coverage.assign(record="coverage")], ignore_index=True)
        limits = (0, max(float(np.nanmax(age)), float(np.median(np.diff(hours)))))
        cfg["value_label"] = "Hours since last observed occupancy (0 = occupied)"
        subtitle = "Time since last tracked occupancy."
        auxiliary["coverage.csv"] = coverage
        foot += "Denominator: pixels occupied at least once during this recording. Grey: no known prior occupancy. "
        foot += "Elapsed time since last occupancy is not an exact vacancy duration. This does not establish rhythmicity or compensation."
    else:
        raise ValueError(kind)
    if kind in {"period", "timing"}:
        full_names = {"lomb": "Lomb-Scargle periodogram", "fft_nlls": "Fourier-initialised nonlinear least squares"}
        method_name = str(params["period_estimation_method"])
        method = full_names.get(method_name, workbench.PERIOD_METHODS[method_name]["label"])
        test = str(params["primary_rhythm_test"])
        test_label = full_names.get(test, workbench.PERIOD_METHODS[test]["label"])
        detrend_label = str(params["detrend"]).replace("_", " ")
        correction = {"bh": "Benjamini-Hochberg", "none": "no"}.get(
            str(params["multiple_testing"]), str(params["multiple_testing"]))
        low, high = params["period_search_hours"]
        foot += (f"Period: {method}, {low:g}-{high:g} h; "
                 f"test: {test_label}; {detrend_label} baseline; "
                 f"{correction} correction across all selected cell-metric tests, "
                 f"threshold {params['rhythmic_alpha']:g}; at least {params['min_cycles_for_confident_period']:g} cycles for support. "
                 "Significance concerns the trace, not a separately fitted component.")
    cfg.update(vmin=limits[0], vmax=limits[1],
                cmap=_get(ctx, "spatial_cmap") or ("magma" if kind == "gaps" else "RdBu_r"))
    if limits[0] == limits[1]:
        cfg["vmax"] = limits[0] + 1
    if kind in {"expansion", "progression", "gaps"}:
        cfg["base_subtitle"] = subtitle
        subtitle += "\n" + ("Full recording shown." if kind == "progression" and cfg["show_history"] else selected_label)
    text = figure_text(ctx.run, ctx.spec.slug, argv=ctx.argv, item=ctx.item,
                       title=title, subtitle=subtitle, footnote=foot, note="")
    canvas_text = text.on_canvas()
    cfg.update(title=canvas_text.title, subtitle=canvas_text.subtitle,
               footnote=canvas_text.footnote, note=canvas_text.note)
    if kind == "period":
        cfg["base_track_footnote"] = canvas_text.footnote.replace(
            cfg.get("track_note", ""), "")
    cfg["placements"] = {}
    for name, default in {"header_x": .055, "title_y": .015, "subtitle_y": .055, "footnote_y": .018}.items():
        placement = _placement(ctx, name, Length(default, "page"))
        cfg["placements"][name] = {"value": placement.value, "unit": placement.unit}
    scripts = Path.home() / ".claude/skills/plot-that/scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from plot_style import apply
    apply()
    ctx.theme.apply()
    cfg["rc"] = {name: mpl.rcParams[name] for name in (
        "font.family", "font.sans-serif", "font.size", "axes.labelsize", "axes.titlesize",
        "axes.linewidth", "xtick.labelsize", "ytick.labelsize", "xtick.major.width",
        "ytick.major.width", "svg.fonttype", "figure.facecolor", "savefig.facecolor")}
    cfg = _safe(cfg)
    figure = spatial.render(data, cfg)
    master = ctx.name.replace("/", "-").replace("\\", "-") + ".svg"
    producer = f'''"""Exact spatial figure reproduction from the bundled encodings."""
from pathlib import Path
import argparse
import pandas as pd
import reprofig
import src_renderer

CONFIG = {cfg!r}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    target = args.output or bundle / "reproduced.svg"
    data = pd.read_csv(bundle / "figure_data.csv")
    fig = src_renderer.render(data, CONFIG)
    record = reprofig.extract_record(bundle / {master!r})
    reprofig.save_figure(fig, target, record=record,
                         savefig_kwargs={{"bbox_inches": "tight", "transparent": True}})

if __name__ == "__main__":
    main()
'''
    settings = {name: _get(ctx, name) for name in [opt.name for opt in ctx.spec.options]}
    auxiliary["settings.csv"] = pd.DataFrame([{"settings_json": json.dumps(_safe(settings)),
                                              "analysis_params_json": json.dumps(_safe(params))}])
    ctx.record_source("manifest.json", ctx.run / "manifest.json")
    readme = (f"# {text.title}\n\n{text.subtitle}\n\n## Interpretation\n\n{text.footnote}\n\n"
              "## Reuse\n\nThis is an additional figure; tissue tectonics is unchanged. "
              "The generic renderer accepts the exact encoding table without running rhythm tests. "
              "Metric selection and all Circadian Workbench detrending controls are exposed through the "
              "figure registry and main analysis plot plans. The source traces, full per-trace test results "
              "when applicable, resolved settings and copied source fingerprints accompany the figure.\n\n"
              "`python plot.py --output reproduced.svg` regenerates the same panels from `figure_data.csv`. "
              "No original run folder or Motion installation is needed for this display-only reproduction. "
              "Re-analysis uses the registered project builder and copied measurement sources.\n")
    return FigureResult(figure=figure, axes=figure.axes, figure_data=data,
                         subtitle=text.subtitle, footnote=text.footnote,
                         heading=text.claim or ctx.spec.summary, readme=readme, auxiliary=auxiliary,
                         standalone_producer=producer,
                         producer_sources={"renderer.py": Path(spatial.__file__),
                                           "measurements.py": Path(measurements.__file__),
                                           "builder.py": Path(__file__), "entry.py": ctx.spec.source})
