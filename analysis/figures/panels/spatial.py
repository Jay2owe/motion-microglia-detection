"""Generic spatial marks and complete, data-only spatial layouts.

This module deliberately imports no Motion measurement or rhythm code. A copied
module and the exact encoding table reproduce a figure without the source run.
"""
from __future__ import annotations

import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


PHASE_CYCLE = LinearSegmentedColormap.from_list(
    "phase_cycle",
    ["#4878A8", "#0e8f8f", "#d98a17", "#c0392b", "#4878A8"],
    N=256,
)


def _colormap(cmap):
    """Resolve the local cyclic phase palette or a Matplotlib palette name."""
    return PHASE_CYCLE if cmap == PHASE_CYCLE.name else cmap


def field_map(ax, pixels, *, shape, scale=1., cmap="RdBu_r", vmin=-2., vmax=2.):
    """Paint an arbitrary scalar per image pixel; missing is not zero."""
    arr = np.full(shape, np.nan)
    if len(pixels):
        arr[pixels.y.astype(int), pixels.x.astype(int)] = pixels.value.to_numpy(float)
    colors = plt.get_cmap(_colormap(cmap)).copy()
    colors.set_bad("#eeeeee")
    handle = ax.imshow(arr, cmap=colors, vmin=vmin, vmax=vmax,
                       extent=(0, shape[1] * scale, shape[0] * scale, 0),
                       interpolation="nearest", aspect="equal")
    return handle


def cell_map(ax, cells, *, value="value", cmap="viridis", vmin=0, vmax=1,
             point_size=160, annotate=False):
    """One mark per cell, retaining unavailable results as distinct symbols."""
    finite = np.isfinite(pd.to_numeric(cells[value], errors="coerce"))
    status = cells.get("display_status", pd.Series("", index=cells.index))
    no = ~finite & status.eq("not significant")
    other = ~finite & ~no
    ax.scatter(cells.loc[no, "x"], cells.loc[no, "y"], s=point_size,
               c="#999999", edgecolors="white", linewidths=.6, zorder=3)
    ax.scatter(cells.loc[other, "x"], cells.loc[other, "y"], s=point_size,
               facecolors="none", edgecolors="#555555", linewidths=1.2, zorder=3)
    handle = ax.scatter(cells.loc[finite, "x"], cells.loc[finite, "y"],
                        c=cells.loc[finite, value], s=point_size, cmap=_colormap(cmap),
                        vmin=vmin, vmax=vmax, edgecolors="white", linewidths=.6, zorder=4)
    exploratory = finite & status.eq("exploratory period")
    ax.scatter(cells.loc[exploratory, "x"], cells.loc[exploratory, "y"],
               s=point_size * 1.45, facecolors="none", edgecolors="black", linewidths=1, zorder=5)
    if annotate:
        for row in cells.itertuples():
            ax.annotate(str(int(row.identity)), (row.x, row.y), xytext=(4, 4),
                        textcoords="offset points", fontsize=9)
    return handle


def spatial_matrix(ax, traces, *, order, hours, cmap="RdBu_r", vmin=-2, vmax=2,
                   discrete=False):
    """Cell-by-time matrix; discrete snapshots do not imply data between frames."""
    matrix = traces.pivot(index="identity", columns="hours", values="value").reindex(
        index=order, columns=hours)
    times = np.asarray(hours, float)
    half = np.diff(times) / 2
    edges = (np.r_[times[0] - half[0], times[:-1] + half, times[-1] + half[-1]]
             if len(times) > 1 else np.array([times[0] - .25, times[0] + .25]))
    colors = plt.get_cmap(_colormap(cmap)).copy()
    colors.set_bad("#dddddd")
    if discrete:
        edges = np.arange(len(times) + 1) - .5
        ax.set_xticks(np.arange(len(times)), [f"{time:g}" for time in times])
    handle = ax.pcolormesh(edges, np.arange(len(order) + 1) - .5,
                           np.ma.masked_invalid(matrix.to_numpy(float)),
                           cmap=colors, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_ylim(len(order) - .5, -.5)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(int(x)) for x in order], fontsize=9)
    return handle


def connection_map(ax, edges, *, cmap="RdBu_r", vmin=-1, vmax=1, linewidth=2):
    """Arbitrary scalar on supplied spatial edges; no edge selection or tests."""
    finite = edges[np.isfinite(edges.value)]
    segments = np.stack([finite[["x", "y"]], finite[["x2", "y2"]]], axis=1)
    handle = LineCollection(segments, array=finite.value.to_numpy(float),
                            cmap=_colormap(cmap), norm=Normalize(vmin, vmax), linewidths=linewidth)
    ax.add_collection(handle)
    return handle


def track_map(ax, segments, *, cmap="viridis", vmin=0, vmax=1, linewidth=1.6):
    """Generic preselected path segments, with missing scalar values in grey."""
    if not np.isfinite(linewidth) or linewidth <= 0:
        raise ValueError("track line width must be finite and positive")
    if segments.empty:
        handle = LineCollection([], linewidths=linewidth, zorder=2)
        ax.add_collection(handle)
        return handle
    valid = np.isfinite(segments[["x", "y", "x2", "y2"]]).all(axis=1)
    rows = segments[valid]
    lines = np.stack([rows[["x", "y"]], rows[["x2", "y2"]]], axis=1)
    values = pd.to_numeric(rows.value, errors="coerce").to_numpy(float)
    colours = plt.get_cmap(_colormap(cmap))(
        Normalize(vmin, vmax)(np.nan_to_num(values, nan=vmin)))
    colours[~np.isfinite(values)] = to_rgba("#999999")
    handle = LineCollection(lines, colors=colours, linewidths=linewidth, zorder=2)
    ax.add_collection(handle)
    return handle


def _map_axes(ax, config):
    ax.set_xlim(0, config["shape"][1] * config["scale"])
    ax.set_ylim(config["shape"][0] * config["scale"], 0)
    ax.set_aspect("equal")
    ax.set_facecolor("#eeeeee")
    ax.set_xlabel(f"X position ({config['length_unit']})", fontsize=18)
    ax.set_ylabel(f"Y position ({config['length_unit']})", fontsize=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=14)


def _bar(fig, ax, handle, label, *, ticks=None):
    # Fixed readable shape shared by every spatial plot; no thumbnail keys.
    bar = fig.colorbar(handle, ax=ax, fraction=.047, pad=.045, shrink=.75, aspect=18)
    bar.set_label(textwrap.fill(label, 36), fontsize=15)
    bar.ax.tick_params(labelsize=13)
    if ticks is not None:
        bar.set_ticks(ticks)
    return bar


def _status_key(fig):
    handles = [Line2D([], [], marker="o", linestyle="", color="#999999", markersize=10,
                      label="Tested; not significant"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                      color="#555555", markersize=10, label="Unavailable / excluded from this view"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor="#bbbbbb",
                      color="black", markersize=10, label="Outlined colour: exploratory period")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .15),
               ncol=3, frameon=False, fontsize=12)


def _fixed_bar(fig, handle, *, x, y, height, label, ticks=None):
    width, full_height = fig.get_size_inches()
    ax = fig.add_axes([x / width, y / full_height, .18 / width, height / full_height])
    bar = fig.colorbar(handle, cax=ax)
    bar.set_label(textwrap.fill(label, 38), fontsize=14)
    bar.ax.tick_params(labelsize=13)
    if ticks is not None:
        bar.set_ticks(ticks)
    return bar


def render(data: pd.DataFrame, config: dict):
    """Draw only supplied encodings. Tests and metric choices happen upstream."""
    with plt.rc_context(config.get("rc", {})):
        return _render(data, config)


def _render(data, cfg):
    kind = cfg["kind"]
    size = cfg.get("panel_inches", 6.)
    if kind in {"expansion", "gaps"}:
        # Explicit snapshot metadata retains even a completely empty final frame.
        snapshots = cfg.get("snapshots")
        if snapshots is None:
            snapshots = data.loc[data.record.eq("pixel"), ["tile", "hours"]].drop_duplicates().sort_values("hours").to_dict("records")
        if not snapshots:
            raise ValueError("a spatial map needs at least one declared snapshot")
        tiles = [item["tile"] for item in snapshots]
        cols = min(2, len(tiles))
        rows = int(np.ceil(len(tiles) / cols))
        history = cfg.get("show_history", False)
        extra = 4.2 if kind == "gaps" and history else 0
        tile_height = size * cfg["shape"][0] / cfg["shape"][1]
        width = 1.1 + cols * size + (cols - 1) * 1.0 + 2.0
        height = 3.4 + rows * (tile_height + 1.3) + extra
        fig = plt.figure(figsize=(width, height))
        for j, tile in enumerate(tiles):
            bottom = height - 2.2 - tile_height - (j // cols) * (tile_height + 1.3)
            left = 1.1 + (j % cols) * (size + 1.)
            ax = fig.add_axes([left / width, bottom / height, size / width, tile_height / height])
            pixels = data[data.record.eq("pixel") & data.tile.eq(tile)]
            handle = field_map(ax, pixels, shape=cfg["shape"], scale=cfg["scale"],
                               cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"])
            _map_axes(ax, cfg)
            hour = float(snapshots[j]["hours"])
            ax.set_title(f"{hour:g} h", fontsize=40, fontweight="bold", pad=6)
            if j % cols:
                ax.set_ylabel("")
            arrows = data[data.record.eq("arrow") & data.tile.eq(tile)]
            if len(arrows):
                ax.quiver(arrows.x, arrows.y, arrows.x2 - arrows.x, arrows.y2 - arrows.y,
                          angles="xy", scale_units="xy", scale=1, color="black", width=.004)
        _fixed_bar(fig, handle, x=width - 1.65, y=height - 2.2 - tile_height,
                    height=tile_height, label=cfg["value_label"])
        if kind == "gaps" and history:
            ax = fig.add_axes([1.1 / width, 1.8 / height, (width - 3.1) / width, 2.4 / height])
            rows = data[data.record.eq("coverage")].sort_values("hours")
            ax.plot(rows.hours, rows.value * 100, color=cfg["line_color"], linewidth=2.5)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xticks(cfg["hour_ticks"])
            ax.set_xlabel("Hours from start of recording", fontsize=18)
            ax.set_ylabel("Occupied share of\never-occupied tissue (%)", fontsize=18)
            ax.spines[["top", "right"]].set_visible(False)
        elif kind == "gaps":
            coverage = data[data.record.eq("coverage")].set_index("hours")
            summary = "; ".join(f"{item['hours']:g} h: {coverage.loc[item['hours'], 'value'] * 100:.1f}%"
                                for item in snapshots if item["hours"] in coverage.index)
            fig.text(1.1 / width, 1.9 / height, textwrap.fill("Occupied share of ever-occupied tissue: " + summary, int(width * 7)),
                     fontsize=13, va="top")
    elif kind == "period":
        tiles = cfg["tiles"]
        cols = min(2, len(tiles)); rows = int(np.ceil(len(tiles) / cols))
        tile_height = size * cfg["shape"][0] / cfg["shape"][1]
        width = 1.1 + cols * (size + 2.7)
        height = 4.8 + rows * (tile_height + 1.3)
        fig = plt.figure(figsize=(width, height))
        for j, tile in enumerate(tiles):
            left = 1.1 + (j % cols) * (size + 2.7)
            bottom = height - 2.7 - tile_height - (j // cols) * (tile_height + 1.3)
            ax = fig.add_axes([left / width, bottom / height, size / width, tile_height / height])
            cells = data[data.record.eq("cell") & data.tile.eq(tile["key"])].copy()
            if tile["key"] == "period" and cfg.get("spatial_tracks", True):
                tracks = data[data.record.eq("track") & data.tile.eq("period")]
                track_map(ax, tracks, cmap=tile["cmap"], vmin=tile["vmin"], vmax=tile["vmax"],
                          linewidth=cfg.get("spatial_track_width", 1.6))
            handle = cell_map(ax, cells, cmap=tile["cmap"], vmin=tile["vmin"], vmax=tile["vmax"],
                              point_size=cfg["point_size"], annotate=cfg["annotate"])
            _map_axes(ax, cfg)
            ax.set_title(tile["title"], fontsize=20, fontweight="bold", pad=10)
            _fixed_bar(fig, handle, x=left + size + .2, y=bottom, height=tile_height,
                       label=tile["label"], ticks=np.linspace(tile["vmin"], tile["vmax"], 5))
        _status_key(fig)
    elif kind == "progression":
        cells = data[data.record.eq("cell")].sort_values(["spatial_order", "identity"])
        traces = data[data.record.eq("trace")]
        height = max(10, .17 * len(cells) + 3.4)
        width = size + 15
        tile_height = size * cfg["shape"][0] / cfg["shape"][1]
        fig = plt.figure(figsize=(width, height))
        axes = [fig.add_axes([1.1 / width, (height - tile_height) / 2 / height,
                              size / width, tile_height / height]),
                fig.add_axes([(size + 3.1) / width, .15, 8.5 / width, .72])]
        cell_map(axes[0], cells, value="spatial_order", cmap="viridis", vmin=1, vmax=max(2, len(cells)),
                 point_size=cfg["point_size"], annotate=cfg["annotate"])
        _map_axes(axes[0], cfg)
        axes[0].set_title(f"Order along {cfg['spatial_axis'].upper()} position", fontweight="bold")
        handle = spatial_matrix(axes[1], traces, order=cells.identity.tolist(),
                                hours=cfg["hours"], cmap=cfg["cmap"],
                                vmin=cfg["vmin"], vmax=cfg["vmax"],
                                discrete=not cfg.get("show_history", False))
        if cfg.get("show_history", False):
            axes[1].set_xticks(cfg["hour_ticks"])
        axes[1].set_xlabel("Hours from start of recording" if cfg.get("show_history", False)
                           else "Selected recording hours (separate snapshots)")
        axes[1].set_ylabel("Cell identity, in spatial order")
        axes[1].set_title("Measured changes through time" if cfg.get("show_history", False)
                          else ("Final recorded state" if cfg["hours"] == [cfg.get("recording_hours", cfg["hours"])[-1]]
                                else "Values at selected frames"), fontweight="bold")
        _fixed_bar(fig, handle, x=width - 2.6, y=height * .40, height=min(5, height * .4),
                   label=cfg["value_label"])
    elif kind == "timing":
        tile_height = size * cfg["shape"][0] / cfg["shape"][1]
        width, height = size + 3.8, tile_height + 5.8
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes([1.1 / width, 3.8 / height, size / width, tile_height / height])
        handle = cell_map(ax, data, cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"],
                          point_size=cfg["point_size"], annotate=cfg["annotate"])
        _map_axes(ax, cfg)
        if np.isfinite(data.value).any():
            _fixed_bar(fig, handle, x=size + 1.35, y=3.8, height=tile_height,
                       label="Peak delay (h): positive = selected metric peaks later")
        else:
            ax.text(.5, 1.035, "No cells met both rhythm and period-compatibility criteria",
                    transform=ax.transAxes, ha="center", fontsize=12)
        fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                    color="#555555", markersize=10, label="Not comparable / insufficient rhythm evidence")],
                   loc="lower center", bbox_to_anchor=(.5, 2.75 / height), frameon=False, fontsize=12)
    elif kind == "neighbours":
        tile_height = size * cfg["shape"][0] / cfg["shape"][1]
        width, height = size * 2 + 6., tile_height + 4.8
        fig = plt.figure(figsize=(width, height))
        axes = [fig.add_axes([1.1 / width, 2.4 / height, size / width, tile_height / height]),
                fig.add_axes([(size + 4.6) / width, 2.4 / height, size / width, tile_height / height])]
        pairs = data[data.record.eq("pair")]
        edges = pairs[pairs.neighbour.astype(str).str.lower().isin(["true", "1", "1.0"])]
        handle = connection_map(axes[0], edges, cmap=cfg["cmap"])
        nodes = data[data.record.eq("cell")]
        axes[0].scatter(nodes.x, nodes.y, c="#303030", s=45, zorder=4)
        _map_axes(axes[0], cfg)
        axes[0].set_title("Nearby cells: simultaneous change", fontsize=18, fontweight="bold")
        _fixed_bar(fig, handle, x=size + 1.35, y=2.4, height=tile_height,
                   label="Correlation: -1 opposite, +1 together", ticks=[-1, -.5, 0, .5, 1])
        bins = data[data.record.eq("null_bin")]
        axes[1].bar(bins.x, bins.value, width=bins.x2 - bins.x, align="edge", color="#bbbbbb")
        effect = cfg["contrast"]
        if effect is not None:
            axes[1].axvline(effect, color=cfg["line_color"], linewidth=3, label="Measured difference")
        axes[1].set_xlabel("Mean neighbour correlation minus\nmean non-neighbour correlation", fontsize=17)
        axes[1].set_ylabel("Shuffled spatial arrangements", fontsize=17)
        axes[1].set_title(cfg["test_label"], fontsize=17, fontweight="bold")
        axes[1].legend(frameon=False, fontsize=13)
        axes[1].spines[["top", "right"]].set_visible(False)
    else:
        raise ValueError(f"unknown spatial layout {kind!r}")
    placements = cfg.get("placements", {})
    def fraction(name, default, span):
        value = placements.get(name, {"value": default, "unit": "page"})
        return value["value"] / span if value["unit"] == "in" else value["value"]
    x = fraction("header_x", .055, fig.get_figwidth())
    title_y = 1 - fraction("title_y", .015, fig.get_figheight())
    subtitle_y = 1 - fraction("subtitle_y", .055, fig.get_figheight())
    footnote_y = fraction("footnote_y", .018, fig.get_figheight())
    fig.suptitle(cfg["title"], x=x, y=title_y, ha="left", fontsize=25, fontweight="bold")
    fig.text(x, subtitle_y, textwrap.fill(cfg["subtitle"], max(65, int(fig.get_figwidth() * 8))),
             va="top", fontsize=15)
    footnote = cfg["footnote"] + ("\n" + cfg["note"] if cfg.get("note") else "")
    fig.text(x, footnote_y, textwrap.fill(footnote, max(65, int(fig.get_figwidth() * 9))),
             va="bottom", fontsize=12, color="#444444")
    return fig
