"""Panels for what the ``surveillance`` module measures.

Surveillance is a cell rebuilding itself in place: the outline changes shape
faster than the cell goes anywhere. Two of its figures need no panel of their
own - a cell's turnover against its movement is a plain scatter with its
margins, which is :func:`common.scatter_with_margins`, and a turnover trace is
:func:`common.trace`. A panel earns a place here only when it has to know
something about the measurement.

:func:`evidence_channels` does. The tracker scores every pair of consecutive
frames per pixel and writes the result as its own image stack, and the three
channels have colours already - the ones the stack itself encodes them with. A
reader who has watched the quality-control movie has learned green-red-blue;
recolouring them on a figure is a second vocabulary for one thing, so the
channel names carry their roles here rather than being chosen per figure.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import common
from ._contract import PanelResult

__all__ = [
    "EVIDENCE_CHANNELS", "evidence_channels", "exchange_ledger",
    "cancelled_fraction", "mirrored_ledger", "boundary_reference",
    "upheaval_raster", "breakout_events",
]

#: column -> (colour role, what to call it). The default three: what stayed,
#: what arrived and what left. ``base`` and ``trail`` come out of the same stack
#: but answer a different question, so they stay in the table and off the axes.
EVIDENCE_CHANNELS: dict[str, tuple[str, str]] = {
    "frame_held_px": ("evidence_held", "held - structure present on both sides"),
    "frame_gained_px": ("evidence_gained", "gained - where signal arrived"),
    "frame_lost_px": ("evidence_lost", "lost - where signal left"),
    "frame_base_px": ("evidence", "base - the cell body either side"),
    "frame_trail_px": ("evidence", "trail - the path a moving edge swept"),
}


def evidence_channels(
    ax: Any,
    hours: Sequence[float],
    frame: Any,
    theme: Any,
    *,
    columns: Sequence[str],
    looks: dict[str, common.Look] | None = None,
    hour_ticks: float | None = None,
    x_label: str = "Hours from start of recording",
    y_label: str = "Pixels in the transition",
) -> PanelResult:
    """The tracker's motion-evidence channels, one line each, on one axis.

    On one axis deliberately. Signal that leaves one place arrives in another,
    so gained and lost should track each other across the recording; a
    transition where one runs away from the other is one where evidence is being
    created or destroyed rather than moved, and that is only visible when they
    share a scale.
    """
    hours = np.asarray(hours, dtype=float)
    looks = looks or {}

    for column in columns:
        role, label = EVIDENCE_CHANNELS.get(column, ("evidence", column))
        look = looks.get(column) or common.Look(colour=theme.colour(role))
        ax.plot(
            hours, np.asarray(frame[column], dtype=float),
            color=look.colour or theme.colour(role),
            linewidth=theme.stroke("emphasis"), label=label, solid_capstyle="round",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(float(hours.min()), float(hours.max()))
    ax.set_ylim(0, None)
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    drawn = pd.DataFrame({"hours": hours})
    for column in columns:
        drawn[column] = np.asarray(frame[column], dtype=float)
    return PanelResult(data=drawn, axes=ax)


def _ledger_table(
    hours: Sequence[float], frame: pd.DataFrame, gross_column: str,
    lost_column: str, area_change_column: str,
) -> pd.DataFrame:
    table = frame.copy()
    table["hours"] = np.asarray(hours, dtype=float)
    table["gross_px"] = table[gross_column].astype(float) + table[lost_column].astype(float)
    table["net_px"] = table[area_change_column].astype(float)
    table["abs_net_px"] = table["net_px"].abs()
    table["cancelled_fraction"] = np.where(
        table["gross_px"] > 0,
        1.0 - table["abs_net_px"] / table["gross_px"],
        np.nan,
    )
    keep = [column for column in ("identity", "frame_index", "hours", "gross_px",
                                   "net_px", "abs_net_px", "cancelled_fraction")
            if column in table]
    return table[keep]


def exchange_ledger(
    ax: Any,
    hours: Sequence[float],
    frame: pd.DataFrame,
    theme: Any,
    *,
    gross_column: str = "gained_px",
    lost_column: str = "lost_px",
    area_change_column: str = "area_change_px",
    aggregate: str = "median",
    band: tuple[float, float] = (0.25, 0.75),
    per_cell: bool = True,
    hour_ticks: float | None = None,
    y_label: str = "",
) -> PanelResult:
    """Gross exchange and absolute net change on one shared axis."""
    table = _ledger_table(hours, frame, gross_column, lost_column, area_change_column)
    if per_cell and "identity" in table:
        for _, group in table.groupby("identity"):
            ax.plot(group["hours"], group["gross_px"], color=theme.colour("surveillance"),
                    alpha=0.10, linewidth=theme.stroke("guide"))
            ax.plot(group["hours"], group["abs_net_px"], color=theme.colour("morphology"),
                    alpha=0.10, linewidth=theme.stroke("guide"))
    grouped = table.groupby("hours")
    reducer = grouped.median if aggregate == "median" else grouped.mean
    summary = reducer(numeric_only=True)
    low = grouped[["gross_px", "abs_net_px"]].quantile(float(band[0]))
    high = grouped[["gross_px", "abs_net_px"]].quantile(float(band[1]))
    for column, role, label in (
        ("gross_px", "surveillance", "Gross exchange: gained + lost area"),
        ("abs_net_px", "morphology", "Absolute net size change"),
    ):
        ax.fill_between(summary.index, low[column], high[column],
                        color=theme.colour(role), alpha=0.16, linewidth=0)
        ax.plot(summary.index, summary[column], color=theme.colour(role),
                linewidth=theme.stroke("emphasis"), label=label)
    ax.set_xlim(float(table["hours"].min()), float(table["hours"].max()))
    ax.set_xticks(theme.hour_ticks(float(table["hours"].min()), float(table["hours"].max()), hour_ticks))
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel(y_label)
    return PanelResult(data=table, axes=ax)


def cancelled_fraction(
    ax: Any,
    hours: Sequence[float],
    frame: pd.DataFrame,
    theme: Any,
    *,
    per_cell: bool = True,
    aggregate: str = "median",
    marks: Sequence[common.Mark] = (),
    hour_ticks: float | None = None,
) -> PanelResult:
    """The share of gross exchange cancelled by opposite movement."""
    table = frame.copy()
    table["hours"] = np.asarray(hours, dtype=float)
    if "cancelled_fraction" not in table:
        table = _ledger_table(table["hours"], table, "gained_px", "lost_px", "area_change_px")
    if per_cell and "identity" in table:
        for _, group in table.groupby("identity"):
            ax.plot(group["hours"], group["cancelled_fraction"],
                    color=theme.colour("surveillance"), alpha=0.10,
                    linewidth=theme.stroke("guide"))
    grouped = table.groupby("hours")["cancelled_fraction"]
    summary = grouped.median() if aggregate == "median" else grouped.mean()
    ax.plot(summary.index, summary.values, color=theme.colour("surveillance"),
            linewidth=theme.stroke("emphasis"))
    if marks:
        common.reference_lines(ax, marks, theme, orientation="horizontal")
    ax.set_ylim(0, 1)
    ax.set_xlim(float(table["hours"].min()), float(table["hours"].max()))
    ax.set_xticks(theme.hour_ticks(float(table["hours"].min()), float(table["hours"].max()), hour_ticks))
    ax.set_ylabel("Exchange cancelled by\nopposing edge motion")
    kept = [column for column in
            ("identity", "frame_index", "hours", "cancelled_fraction")
            if column in table]
    return PanelResult(data=table[kept], axes=ax)


def mirrored_ledger(
    ax: Any,
    hours: Sequence[float],
    frame: pd.DataFrame,
    theme: Any,
    *,
    gained_column: str = "gained_px",
    lost_column: str = "lost_px",
    held_column: str = "held_px",
    net_column: str = "area_change_px",
    show_held: bool = True,
    hour_ticks: float | None = None,
    y_label: str = "Pixels in the transition",
) -> PanelResult:
    """Gained above zero, lost below it, with held and net for scale."""
    hours = np.asarray(hours, dtype=float)
    gained = np.asarray(frame[gained_column], dtype=float)
    lost = -np.asarray(frame[lost_column], dtype=float)
    net = np.asarray(frame[net_column], dtype=float)
    ax.fill_between(hours, 0, gained, color=theme.colour("evidence_gained"), alpha=0.35,
                    label="Area gained")
    ax.fill_between(hours, 0, lost, color=theme.colour("evidence_lost"), alpha=0.35,
                    label="Area lost")
    if show_held and held_column in frame:
        held = np.asarray(frame[held_column], dtype=float)
        ax.plot(hours, held, color=theme.colour("reference"), alpha=0.65,
                linewidth=theme.stroke("guide"), label="Area retained")
    ax.plot(hours, net, color=theme.colour("ink"), linewidth=theme.stroke("emphasis"),
            label="Net size change")
    ax.axhline(0, color=theme.colour("ink"), linewidth=theme.stroke("guide"))
    ax.set_xlim(float(hours.min()), float(hours.max()))
    ax.set_xticks(theme.hour_ticks(float(hours.min()), float(hours.max()), hour_ticks))
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel(y_label)
    result = pd.DataFrame({
        "hours": hours, "gained_px": gained, "lost_px": -lost,
        "held_px": np.asarray(frame[held_column], dtype=float) if held_column in frame else np.nan,
        "net_px": net,
    })
    if "identity" in frame:
        result.insert(0, "identity", frame["identity"].to_numpy())
    return PanelResult(data=result, axes=ax)


def boundary_reference(
    ax: Any,
    perimeter: Sequence[float],
    theme: Any,
    *,
    pixels: Sequence[float] = (1.0, 2.0),
    role: str = "reference",
    label_at: float = 0.8,
) -> PanelResult:
    """Reference lines for one or more boundary-length movements."""
    x = np.sort(np.asarray(perimeter, dtype=float))
    curves = []
    for index, movement in enumerate(pixels):
        y = x * float(movement)
        curves.append(y)
        ax.plot(x, y, color=theme.colour(role), linewidth=theme.stroke("guide"),
                linestyle=(0, (5, 3)))
        if len(x):
            at = min(max(int((len(x) - 1) * label_at), 0), len(x) - 1)
            ax.text(x[at], y[at], f"{movement:g} boundary unit", va="bottom",
                    fontsize=theme.size("caption"), color=theme.colour(role))
    drawn = pd.concat(
        [pd.DataFrame({"movement": float(movement), "perimeter": x, "exchange": y})
         for movement, y in zip(pixels, curves)],
        ignore_index=True,
    ) if curves else pd.DataFrame(columns=["movement", "perimeter", "exchange"])
    return PanelResult(data=drawn, axes=ax,
                       extra={"curves": np.asarray(curves)})


def upheaval_raster(
    ax: Any,
    frame: pd.DataFrame,
    theme: Any,
    *,
    jaccard_column: str = "jaccard",
    threshold: float | None = None,
    order: Sequence[Any] | None = None,
    hour_ticks: float | None = None,
    y_label: str = "Cell, ordered by first appearance",
    event_label: str | None = None,
) -> PanelResult:
    """Marks each low-overlap transition; returns those event rows."""
    data = frame.dropna(subset=[jaccard_column, "hours", "identity"]).copy()
    threshold = float(data[jaccard_column].quantile(0.05)) if threshold is None else float(threshold)
    events = data[data[jaccard_column] <= threshold].copy()
    identities = list(order) if order is not None else (
        data.groupby("identity")["hours"].min().sort_values().index.tolist()
    )
    row = {identity: index for index, identity in enumerate(identities)}
    events["row_order"] = events["identity"].map(row)
    ax.scatter(events["hours"], events["row_order"], color=theme.colour("highlight"),
               s=theme.point_area(0.8), marker="|", label="Low-overlap event")
    ax.set_xlim(float(data["hours"].min()), float(data["hours"].max()))
    ax.set_ylim(len(identities) - 0.5, -0.5)
    ax.set_xticks(theme.hour_ticks(float(data["hours"].min()), float(data["hours"].max()), hour_ticks))
    ax.set_xlabel("Hours from start of recording")
    ax.set_ylabel(y_label)
    meaning = event_label or jaccard_column
    ax.text(0.99, 0.98, f"Event: {meaning} ≤ {threshold:.3g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=theme.size("caption"))
    events["event_threshold"] = threshold
    return PanelResult(data=events, axes=ax)


def breakout_events(
    frame: pd.DataFrame,
    *,
    column: str = "cancelled_fraction",
    per_cell: bool = True,
    rank: int = 1,
    window: int = 12,
) -> pd.DataFrame:
    """Align rows on each cell's ranked excursion outside its own IQR."""
    data = frame.copy()
    if column not in data and column == "cancelled_fraction":
        gross = data["gained_px"].astype(float) + data["lost_px"].astype(float)
        data[column] = np.where(gross > 0, 1 - data["area_change_px"].abs() / gross, np.nan)
    groups = data.groupby("identity", sort=True) if per_cell else [("all", data)]
    aligned = []
    for identity, group in groups:
        group = group.sort_values("frame_index")
        values = group[column].astype(float)
        q1, q3 = values.quantile([0.25, 0.75])
        centre = float(values.median())
        scale = float(q3 - q1) or 1.0
        score = (values - centre).abs() / scale
        candidates = score.sort_values(ascending=False)
        if len(candidates) < int(rank):
            continue
        event_index = candidates.index[int(rank) - 1]
        event_frame = int(group.loc[event_index, "frame_index"])
        piece = group[
            group["frame_index"].between(event_frame - int(window), event_frame + int(window))
        ].copy()
        piece["event_frame_index"] = event_frame
        piece["offset_frames"] = piece["frame_index"].astype(int) - event_frame
        piece["event_rank"] = int(rank)
        piece["event_score"] = float(score.loc[event_index])
        aligned.append(piece)
    return pd.concat(aligned, ignore_index=True) if aligned else pd.DataFrame()
