"""Redraw the six spatial bundles without repeating their statistical analyses.

Reads only literal configuration and CSV encodings, never executes a producer
from an arbitrary bundle. The live generic renderer supplies layout fixes.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import tifffile

from _schema import FigureResult, load_all
from _flat_bundle import finish
from panels import spatial
from analysis import spatial as measurements
from _spatial import _pixels, period_track_data

SLUGS = ("tissue-expansion-sequence", "spatial-rhythm-maps", "spatial-rhythm-progression",
         "reporter-shape-timing", "neighbour-coordination", "tissue-coverage-gaps")


def time_view(bundle, data, config, *, snapshot_count=None, snapshot_hours=None, show_history=None):
    """Select a new display from cached measurements, without repeating fits."""
    cfg = dict(config)
    kind = cfg["kind"]
    if kind not in {"expansion", "progression", "gaps"}:
        if any(value is not None for value in (snapshot_count, snapshot_hours, show_history)):
            raise ValueError("snapshot controls apply to time-point plots, not whole-recording statistical summaries")
        return data, cfg, {}
    cfg["recording_hours"] = cfg.get("recording_hours", cfg["hours"])
    hours = np.asarray(cfg["recording_hours"], float)
    count = cfg.get("snapshot_count", 1) if snapshot_count is None else snapshot_count
    # A newly requested count deliberately clears an older explicit selection.
    requested = (cfg.get("snapshot_hours", []) if snapshot_count is None else []) if snapshot_hours is None else snapshot_hours
    indices = measurements.snapshot_indices(hours, requested, count=count)
    cfg.update(snapshot_count=count, snapshot_hours=requested,
               show_history=cfg.get("show_history", False) if show_history is None else show_history)
    cfg["snapshots"] = [dict(tile=tile, frame_index=int(index), hours=float(hours[index]))
                        for tile, index in enumerate(indices)]
    extra = {"der_snapshots.csv": pd.DataFrame(cfg["snapshots"])}
    if kind in {"expansion", "gaps"}:
        labels = tifffile.imread(bundle / "src_labels.tif")
        if labels.shape != (len(hours), *cfg["shape"]):
            raise ValueError("cached labels do not match the recorded time axis and field")
    if kind == "expansion":
        traces = pd.read_csv(bundle / "der_display_traces.csv")
        prior_data = data
        data = _pixels(labels, traces, indices, hours)
        settings = json.loads(pd.read_csv(bundle / "der_settings.csv").settings_json.iloc[0])
        if settings.get("spatial_arrows", False) or prior_data.record.eq("arrow").any():
            centre = settings.get("spatial_centre", "soma")
            frame = pd.read_csv(bundle / "src_cell_frame.csv").sort_values(["identity", "frame_index"])
            previous = frame.groupby("identity").shift()
            for tile, index in enumerate(indices):
                valid = frame.frame_index.eq(index) & (frame.frame_index - previous.frame_index).eq(1)
                current = frame[valid]
                arrows = pd.DataFrame(dict(record="arrow", tile=tile, hours=hours[index],
                    identity=current.identity, x=previous.loc[valid, f"{centre}_x"] * cfg["scale"],
                    y=previous.loc[valid, f"{centre}_y"] * cfg["scale"],
                    x2=current[f"{centre}_x"] * cfg["scale"], y2=current[f"{centre}_y"] * cfg["scale"]))
                data = pd.concat([data, arrows], ignore_index=True)
    elif kind == "progression":
        cells = data[data.record.eq("cell")]
        traces = pd.read_csv(bundle / "der_display_traces.csv")
        if not cfg["show_history"]:
            traces = traces[traces.frame_index.isin(indices)]
        cfg["hours"] = hours.tolist() if cfg["show_history"] else hours[indices].tolist()
        data = pd.concat([cells, traces[["identity", "hours", "value"]].assign(record="trace")], ignore_index=True)
    else:
        domain, age, coverage = measurements.coverage_gaps(labels > 0, hours)
        yy, xx = np.nonzero(domain)
        data = pd.concat([pd.DataFrame(dict(record="pixel", tile=tile, hours=hours[index],
                           x=xx, y=yy, value=age[index, yy, xx])) for tile, index in enumerate(indices)], ignore_index=True)
        shown = coverage if cfg["show_history"] else coverage.iloc[indices]
        data = pd.concat([data, shown.assign(record="coverage")], ignore_index=True)
    view = ("Final recorded frame: " + f"{hours[-1]:g} h." if list(indices) == [len(hours) - 1]
            else "Selected recorded hours: " + ", ".join(f"{hours[i]:g}" for i in indices) + ".")
    # Keep editable scientific wording, but replace only our own generated view line.
    cfg["base_subtitle"] = cfg.get("base_subtitle", "Time since last tracked occupancy." if kind == "gaps" else cfg["subtitle"])
    cfg["subtitle"] = cfg["base_subtitle"] + "\n" + ("Full recording shown." if kind == "progression" and cfg["show_history"] else view)
    if kind == "progression":
        warning = "Repeated diagonal bands indicate sequential timing, not proof of a propagating signal."
        cfg["footnote"] = cfg.get("footnote", "").replace(warning, "").strip()
        if cfg["show_history"]:
            cfg["footnote"] += " " + warning
    settings_path = bundle / "der_settings.csv"
    if settings_path.exists():
        settings_table = pd.read_csv(settings_path)
        settings = json.loads(settings_table.settings_json.iloc[0])
        settings.update(snapshot_count=count, snapshot_hours=requested)
        if kind != "expansion":
            settings["show_history"] = cfg["show_history"]
        settings_table.loc[0, "settings_json"] = json.dumps(settings)
        extra["der_settings.csv"] = settings_table
    return data, cfg, extra


def track_view(bundle, data, config, *, enabled=None, scope=None, width=None):
    """Add/select tracks using copied positions and existing significance values."""
    cfg = dict(config)
    if cfg.get("kind") != "period":
        raise ValueError("period track controls apply only to spatial-rhythm-maps")
    enabled = cfg.get("spatial_tracks", True) if enabled is None else enabled
    scope = cfg.get("spatial_track_cells", "rhythmic") if scope is None else scope
    width = cfg.get("spatial_track_width", 1.6) if width is None else width
    if not isinstance(enabled, bool):
        raise ValueError("spatial_tracks must be true or false")
    if scope not in {"rhythmic", "all"}:
        raise ValueError("spatial_track_cells must be rhythmic or all")
    if not np.isfinite(width) or width <= 0:
        raise ValueError("spatial_track_width must be finite and positive")
    settings_table = pd.read_csv(bundle / "der_settings.csv")
    settings = json.loads(settings_table.settings_json.iloc[0])
    data = data[data.record.ne("track")].copy()
    tracks = pd.DataFrame(columns=["identity", "x", "y", "x2", "y2", "value", "record", "tile"])
    centre = settings.get("spatial_centre", "soma")
    if enabled:
        frame = pd.read_csv(bundle / "src_cell_frame.csv")
        tracks = period_track_data(frame, data, scope=scope, centre=centre, scale=cfg["scale"])
        data = pd.concat([data, tracks], ignore_index=True)
    cfg.update(spatial_tracks=enabled, spatial_track_cells=scope, spatial_track_width=width)
    cfg["base_track_footnote"] = cfg.get("base_track_footnote", cfg.get("footnote", ""))
    cfg["footnote"] = cfg["base_track_footnote"]
    if enabled:
        cfg["footnote"] += (f" Tracks: {scope} cells, recorded {centre} paths; gaps are not connected. "
                            "Track colour is the detected period; grey means no displayed period.")
    settings.update(spatial_tracks=enabled, spatial_track_cells=scope, spatial_track_width=width)
    settings_table.loc[0, "settings_json"] = json.dumps(settings)
    return data, cfg, {"der_tracks.csv": tracks, "der_settings.csv": settings_table}


def refresh(run, slug, *, snapshot_count=None, snapshot_hours=None, show_history=None,
            spatial_tracks=None, spatial_track_cells=None, spatial_track_width=None):
    spec = load_all()[slug]
    bundle = Path(run) / "figures" / slug
    producer = (bundle / "plot.py").read_text(encoding="utf-8")
    config_nodes = [node.value for node in ast.parse(producer).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets)]
    if len(config_nodes) != 1:
        raise ValueError("the producer must declare one literal CONFIG")
    config = ast.literal_eval(config_nodes[0])
    data = pd.read_csv(bundle / "figure_data.csv")
    updated = {}
    if config.get("kind") in {"expansion", "progression", "gaps"} or any(
            value is not None for value in (snapshot_count, snapshot_hours, show_history)):
        data, config, updated = time_view(bundle, data, config, snapshot_count=snapshot_count,
                                        snapshot_hours=snapshot_hours, show_history=show_history)
    if config.get("kind") == "period" or any(value is not None for value in
            (spatial_tracks, spatial_track_cells, spatial_track_width)):
        data, config, track_updates = track_view(bundle, data, config, enabled=spatial_tracks,
                                                scope=spatial_track_cells, width=spatial_track_width)
        updated.update(track_updates)
    if updated:
        node = config_nodes[0]
        lines = producer.splitlines(keepends=True)
        # CONFIG is emitted as one literal assignment. Replace its value only.
        producer = ("".join(lines[:node.lineno - 1]) + "CONFIG = " + repr(config) + "\n"
                    + "".join(lines[node.end_lineno:]))
    figure = spatial.render(data, config)
    index = pd.read_csv(bundle / "sources.csv")
    # Keep the analysed input snapshot: the live source run may have changed
    # since these statistics were computed. Only renderer code is refreshed.
    sources = {row.copied_path.removeprefix("src_"): bundle / row.copied_path for row in index.itertuples()}
    origins = {row.copied_path.removeprefix("src_"): row.original_path for row in index.itertuples()}
    # The exact current renderer is bundled on every layout-only redraw.
    sources["renderer.py"] = Path(spatial.__file__)
    if updated:
        sources["view_selection.py"] = Path(__file__)
        # Preserve original analysis code, but also trace the current selection
        # and occupancy-age calculation used for the revised display.
        sources["snapshot_selection.py"] = Path(measurements.__file__)
        if config.get("kind") == "period":
            sources["track_selection.py"] = Path(__file__).with_name("_spatial.py")
    auxiliary = {path.name: pd.read_csv(path) for path in bundle.glob("der_*.csv")}
    auxiliary.update(updated)
    if (bundle / "statistics.csv").exists():
        auxiliary["statistics.csv"] = pd.read_csv(bundle / "statistics.csv")
    ctx = SimpleNamespace(run=Path(run), bundle=bundle, name=slug, spec=spec, sources=sources,
                          source_origins=origins)
    readme = (bundle / "README.md").read_text(encoding="utf-8")
    if updated:
        readme = readme.split("\n## Display selection\n")[0]
        readme += ("\n## Display selection\n\n" + config["subtitle"] +
                   "\n\nThe exact plotted table contains the displayed encodings. Full cached measurements remain in the derived tables. "
                   "Display and track selection do not refit baselines or repeat rhythm/permutation tests.\n")
        if config.get("kind") == "period":
            readme += (f"Tracks enabled: {config['spatial_tracks']}; cells: {config['spatial_track_cells']}; "
                       f"line width: {config['spatial_track_width']:g} points. Rhythmic-only selection uses the saved significant verdict. "
                       "Only consecutive recorded positions are connected, on the detected-period panel.\n")
    result = FigureResult(figure=figure, axes=figure.axes, figure_data=data,
                          standalone_producer=producer, heading=spec.summary,
                          readme=readme, auxiliary=auxiliary)
    return finish(ctx, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--plots", nargs="+", choices=SLUGS, default=SLUGS)
    parser.add_argument("--snapshot-count", type=int)
    parser.add_argument("--snapshot-hours", type=lambda value: [float(x) for x in value.split(",")])
    parser.add_argument("--show-history", type=json.loads)
    parser.add_argument("--spatial-tracks", type=json.loads)
    parser.add_argument("--spatial-track-cells", choices=["rhythmic", "all"])
    parser.add_argument("--spatial-track-width", type=float)
    args = parser.parse_args()
    for name in args.plots:
        refresh(args.run, name, snapshot_count=args.snapshot_count,
                snapshot_hours=args.snapshot_hours, show_history=args.show_history,
                spatial_tracks=args.spatial_tracks, spatial_track_cells=args.spatial_track_cells,
                spatial_track_width=args.spatial_track_width)
